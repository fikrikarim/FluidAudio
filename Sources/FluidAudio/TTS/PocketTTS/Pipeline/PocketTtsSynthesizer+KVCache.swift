@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Mutable KV cache state passed through conditioning and generation steps.
    ///
    /// One cache per transformer layer stores the K (key) and V (value) projections
    /// for every processed token. This avoids recomputing K/V for past tokens —
    /// each new step only computes its own K/V, then reads all cached K/V via attention.
    struct KVCacheState {
        /// 6 KV cache arrays, each shaped `[2, 1, kvCacheMaxLen, 16, 64]`:
        ///  - `2`: K and V tensors (index 0 = keys, index 1 = values)
        ///  - `1`: batch size
        ///  - `kvCacheMaxLen` (512): pre-allocated position slots
        ///  - `16`: attention heads
        ///  - `64`: dims per head (16 × 64 = 1024 total)
        var caches: [MLMultiArray]
        /// 6 position counters (one per layer) tracking the next write slot in the cache.
        var positions: [MLMultiArray]
    }

    /// Create an empty KV cache state (all zeros, positions at 0).
    static func emptyKVCacheState() throws -> KVCacheState {
        let layers = PocketTtsConstants.kvCacheLayers
        let shape: [NSNumber] = [
            2, 1, NSNumber(value: PocketTtsConstants.kvCacheMaxLen), 16, 64,
        ]

        var caches: [MLMultiArray] = []
        var positions: [MLMultiArray] = []
        caches.reserveCapacity(layers)
        positions.reserveCapacity(layers)

        for _ in 0..<layers {
            let cache = try MLMultiArray(shape: shape, dataType: .float32)
            let cachePtr = cache.dataPointer.bindMemory(
                to: Float.self, capacity: cache.count)
            cachePtr.initialize(repeating: 0, count: cache.count)
            caches.append(cache)

            let pos = try MLMultiArray(shape: [1], dataType: .float32)
            pos[0] = NSNumber(value: Float(0))
            positions.append(pos)
        }

        return KVCacheState(caches: caches, positions: positions)
    }

    /// Run the conditioning step model for a single token, updating the KV cache in place.
    ///
    /// `cond_step` and `flowlm_step` share the same transformer weights. This function
    /// runs the transformer in "prefill mode": it processes one conditioning token
    /// (voice embedding or text embedding), computes K/V projections, and writes them
    /// into the cache at the current position. No audio is produced.
    static func runCondStep(
        conditioning: MLMultiArray,
        state: inout KVCacheState,
        model: MLModel
    ) async throws {
        var inputDict: [String: Any] = [
            "conditioning": conditioning
        ]

        for i in 0..<PocketTtsConstants.kvCacheLayers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        for i in 0..<PocketTtsConstants.kvCacheLayers {
            guard let newCache = output.featureValue(for: CondStepKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing cond_step cache output: \(CondStepKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: CondStepKeys.positionKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing cond_step position output: \(CondStepKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }
    }

    /// Deep-copy a KV cache state. Required because MLMultiArray is a reference type.
    static func cloneKVCacheState(_ state: KVCacheState) throws -> KVCacheState {
        var newCaches: [MLMultiArray] = []
        var newPositions: [MLMultiArray] = []
        newCaches.reserveCapacity(state.caches.count)
        newPositions.reserveCapacity(state.positions.count)

        for cache in state.caches {
            let copy = try MLMultiArray(shape: cache.shape, dataType: cache.dataType)
            let bytes = cache.count * MemoryLayout<Float>.size
            if bytes > 0 {
                copy.dataPointer.copyMemory(from: cache.dataPointer, byteCount: bytes)
            }
            newCaches.append(copy)
        }
        for pos in state.positions {
            let copy = try MLMultiArray(shape: [1], dataType: .float32)
            copy[0] = pos[0]
            newPositions.append(copy)
        }
        return KVCacheState(caches: newCaches, positions: newPositions)
    }

    /// Prefill the KV cache with voice tokens only (positions 0..<promptLength).
    /// The result can be cached and reused across calls with the same voice.
    static func prefillVoiceKVCache(
        voiceData: PocketTtsVoiceData,
        model: MLModel
    ) async throws -> KVCacheState {
        var state = try emptyKVCacheState()
        let dim = PocketTtsConstants.embeddingDim

        for tokenIdx in 0..<voiceData.promptLength {
            let token = try createConditioningToken(
                from: voiceData.audioPrompt,
                offset: tokenIdx * dim,
                dim: dim
            )
            try await runCondStep(conditioning: token, state: &state, model: model)
        }

        let finalPos = state.positions[0][0].floatValue
        logger.info("Voice KV cache prefilled to position \(Int(finalPos))")

        return state
    }

    /// Prefill the KV cache with voice and text conditioning tokens.
    ///
    /// If `cachedVoiceState` is provided, deep-copies it and skips voice prefill.
    /// Otherwise processes voice tokens first, then text tokens (critical ordering).
    static func prefillKVCache(
        voiceData: PocketTtsVoiceData,
        textEmbeddings: [[Float]],
        model: MLModel,
        cachedVoiceState: KVCacheState? = nil
    ) async throws -> KVCacheState {
        var state: KVCacheState

        if let cached = cachedVoiceState {
            // Fast path: clone the pre-computed voice KV state
            state = try cloneKVCacheState(cached)
        } else {
            // Slow path: compute voice KV from scratch
            state = try emptyKVCacheState()
            let dim = PocketTtsConstants.embeddingDim
            for tokenIdx in 0..<voiceData.promptLength {
                let token = try createConditioningToken(
                    from: voiceData.audioPrompt,
                    offset: tokenIdx * dim,
                    dim: dim
                )
                try await runCondStep(conditioning: token, state: &state, model: model)
            }
        }

        // Text tokens
        let dim = PocketTtsConstants.embeddingDim
        for embedding in textEmbeddings {
            let token = try createConditioningToken(from: embedding, offset: 0, dim: dim)
            try await runCondStep(conditioning: token, state: &state, model: model)
        }

        let finalPos = state.positions[0][0].floatValue
        logger.info("KV cache prefilled to position \(Int(finalPos))\(cachedVoiceState != nil ? " (voice cached)" : "")")

        return state
    }

    /// Create a `[1, 1, 1024]` MLMultiArray from a float slice.
    ///
    /// Shape: batch=1, sequence_length=1 (one token at a time), embedding_dim=1024.
    private static func createConditioningToken(
        from source: [Float], offset: Int, dim: Int
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        source.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            ptr.update(from: base.advanced(by: offset), count: dim)
        }
        return array
    }

    /// Run the generation step model, returning transformer output and EOS logit.
    ///
    /// Same transformer as `cond_step`, now in "generate mode". Takes the previous
    /// audio latent (or NaN for BOS), attends to all cached K/V from conditioning
    /// and prior generation steps, and produces a 1024-d hidden state (for flow_decoder)
    /// plus an EOS logit indicating whether the model is done speaking.
    /// Also writes this step's own K/V into the cache for future steps.
    static func runFlowLMStep(
        sequence: MLMultiArray,
        bosEmb: MLMultiArray,
        state: inout KVCacheState,
        model: MLModel
    ) async throws -> (transformerOut: MLMultiArray, eosLogit: Float) {
        var inputDict: [String: Any] = [
            "sequence": sequence,
            "bos_emb": bosEmb,
        ]

        for i in 0..<PocketTtsConstants.kvCacheLayers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract transformer output
        guard let transformerOut = output.featureValue(for: FlowLMStepKeys.transformerOut)?.multiArrayValue
        else {
            throw PocketTTSError.processingFailed("Missing flowlm_step transformer output")
        }

        // Extract EOS logit
        guard let eosArray = output.featureValue(for: FlowLMStepKeys.eosLogit)?.multiArrayValue
        else {
            throw PocketTTSError.processingFailed("Missing flowlm_step EOS logit")
        }
        let eosLogit = eosArray[0].floatValue

        // Update caches and positions
        for i in 0..<PocketTtsConstants.kvCacheLayers {
            guard
                let newCache = output.featureValue(for: FlowLMStepKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing flowlm_step cache output: \(FlowLMStepKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: FlowLMStepKeys.positionKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing flowlm_step position output: \(FlowLMStepKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }

        return (transformerOut: transformerOut, eosLogit: eosLogit)
    }
}
