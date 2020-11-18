// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_state.hpp"
#include "frontend/quantized_layer_params.hpp"
#include <layer_transform.hpp>
#include <blob_factory.hpp>
#include "preprocessing.hpp"
#include "ie_layouts.h"

namespace  GNAPluginNS {

namespace memory {

    std::string GNAMemoryState::GetName() const {
        return name;
    }

    void GNAMemoryState::Reset() {
        state->Reset();
    }

    InferenceEngine::Precision GNAMemoryState::getPrecision() const {
        InferenceEngine::Precision state_precision;

        if (state->getInput()) {
            state_precision = state->getInput()->precision;
        } else {
            auto element_size = state->elementSizeBytes();
            switch (element_size) {
            case 4:
                state_precision = InferenceEngine::Precision::FP32;
                break;
            case 2:
                state_precision = InferenceEngine::Precision::I16;
                break;
            default:
                THROW_GNA_EXCEPTION << "Incorrect state element size " << element_size <<
                    " to determine precision for MemoryState " << name;
            }
        }

        return state_precision;
    }

    void GNAMemoryState::SetState(InferenceEngine::Blob::Ptr newState) {
        IE_ASSERT(newState != nullptr);

        auto data_ptr = newState->cbuffer().as<void*>();
        IE_ASSERT(data_ptr != nullptr);
        auto data_size = newState->byteSize();
        auto data_elements = data_size / newState->element_size();
        if (ALIGN64(state->reserved_size) != ALIGN64((data_size / (newState->element_size() / state->elementSizeBytes())))) {
            THROW_GNA_EXCEPTION << "Failed to SetState. Sizes of new and old states do not match. ("
                << state->reserved_size << " != " << (newState->element_size() / state->elementSizeBytes()) << ")";
        }

        InferenceEngine::Precision state_precision = getPrecision();
        auto new_state_precision = newState->getTensorDesc().getPrecision();

        if (state->gna_ptr == data_ptr) {
            return;
        }

        if (new_state_precision == state_precision) {
            std::cout << ">> Same precision.. Doing mem copy" << std::endl;
            std::memcpy(state->gna_ptr, data_ptr, data_size);
            return;
        }

        switch (state_precision) {
        case InferenceEngine::Precision::I16: {
            if (new_state_precision == InferenceEngine::Precision::FP32) {
                auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(state->getInput());
                auto scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;

                if ((name =="0") || (name == "2") || (name == "4") || (name == "6") || (name == "8") || (name == "10"))
                    scale_factor = 512;
                else
                    scale_factor = 2048;


                GNAPluginNS::ConvertToInt16(static_cast<int16_t*>(state->gna_ptr),
                    newState->buffer().as<float*>(),
                    1,
                    data_elements,
                    scale_factor);

                std::cout << ">> Converting to FP32. scale factor" << scale_factor << std::endl;
            } else {
                THROW_GNA_EXCEPTION << "Failed to SetState for MemoryState " << name
                    << ". If old state precision is I16 only I16 and FP32 are allowed as new state precisions."
                    << " Old state: " << state_precision << " New state: " << new_state_precision;
            }
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Failed to SetState for MemoryState " << name
                << ". Incorrect new/old precision pair"
                << " Old state: " << state_precision << " New state: " << new_state_precision;
        }
    }

    InferenceEngine::Blob::CPtr GNAMemoryState::GetLastState() const {
        auto elements = state->reserved_size / state->elementSizeBytes();
        InferenceEngine::Precision state_precision = getPrecision();

        if (state->getInput() && state_precision == InferenceEngine::Precision::I16) {
            auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(state->getInput());
            auto scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;

            auto result_blob = make_blob_with_precision(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                InferenceEngine::SizeVector({ 1, elements }),
                InferenceEngine::NC));

            result_blob->allocate();
            auto buffer = result_blob->buffer().as<float*>();
            auto new_gna_ptr = static_cast<int16_t*>(state->gna_ptr);

            for (int i = 0; i < elements; i++) {
                buffer[i] = new_gna_ptr[i] / scale_factor;
            }

            return result_blob;
        } else {
            auto result_blob = make_blob_with_precision(InferenceEngine::TensorDesc(state_precision,
                InferenceEngine::SizeVector({ 1, elements }),
                InferenceEngine::NC));
            result_blob->allocate();
            std::memcpy(state->gna_ptr, result_blob->buffer(), state->reserved_size);

            return result_blob;
        }
    }
}  // namespace memory
}  // namespace GNAPluginNS
