// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <array>
#include <details/ie_exception.hpp>
#include <ios>
#include <iomanip>
#include <map>
#include <ie_algorithm.hpp>
#include <ie_common.h>
#include <ie_precision.hpp>
#include <unistd.h>

#if defined __INTEL_COMPILER || defined _MSC_VER
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include "gna_plugin.hpp"
#include "gna_model_serial.hpp"
#include "serial/headers/latest/gna_model_header.hpp"

using namespace GNAPluginNS;

inline void writeNBytes(const void *ptr, uint32_t size, std::ostream & os) {
    os.write(static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T & obj, std::ostream & os) {
    os.write(reinterpret_cast<const char *>(&obj), sizeof(T));
}

template <class T>
inline void readBits(T & obj, std::istream & is) {
    is.read(reinterpret_cast<char *>(&obj), sizeof(T));
}

inline void readNBytes(void * ptr, uint32_t size, std::istream & is) {
    is.read(reinterpret_cast<char *>(ptr), size);
}

template <int nBits, class T>
inline void readNBits(T & obj, std::istream & is) {
    std::array<uint8_t, nBits / 8> tmp;
    is.read(reinterpret_cast<char *>(&tmp), nBits / 8);

    obj = * reinterpret_cast<T*>(&tmp.front());
}

inline void * offsetToPointer(void * const base, uint64_t offset) {
    return reinterpret_cast<uint8_t *>(base) + offset;
}

template <class T>
inline void readOffset(T & ptr, void *base,  std::istream & is) {
    uint64_t offset = 0ull;
    readBits(offset, is);
    ptr = reinterpret_cast<T>(offsetToPointer(base, offset));
}

inline void writeNBytes(const void *ptr, uint32_t size, int fd) {
    auto ret = write(fd, static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T & obj, int fd) {
    auto ret= write(fd, reinterpret_cast<const char *>(&obj), sizeof(T));
}

template <class T>
inline void readBits(T & obj, int fd) {
    auto ret= read(fd, reinterpret_cast<char *>(&obj), sizeof(T));
}

inline void readNBytes(void * ptr, uint32_t size, int fd) {
    auto ret= read(fd, reinterpret_cast<char *>(ptr), size);
}

template <int nBits, class T>
inline void readNBits(T & obj, int fd) {
    std::array<uint8_t, nBits / 8> tmp;
    auto ret= read(fd, reinterpret_cast<char *>(&tmp), nBits / 8);
    obj = * reinterpret_cast<T*>(&tmp.front());
}

template <class T>
inline void readOffset(T & ptr, void *base,  int fd) {
    uint64_t offset = 0ull;
    readBits(offset, fd);
    ptr = reinterpret_cast<T>(offsetToPointer(base, offset));
}

union {
    uint16_t s;
    uint8_t  c[2];
} constexpr static  LECheck {1};

bool is_little_endian() {
    return LECheck.c[0] == 1;
}

const int gna_header_magic = is_little_endian() ?  0x4d414e47 : 0x474e414d;

GNAPluginNS::HeaderLatest::ModelHeader GNAModelSerial::ReadHeader(std::istream &is) {
    is.exceptions(std::istream::failbit);
    is.seekg(0, is.end);
    auto stream_len = is.tellg();
    if (stream_len == -1) {
        THROW_GNA_EXCEPTION << "Can't open file to import";
    }
    is.seekg(0, is.beg);

    HeaderLatest::ModelHeader header;
    header.version.major = 0u;
    header.version.minor = 0u;
    auto size_of_headers_header = sizeof(HeaderLatest::ModelHeader::gnam) + sizeof(HeaderLatest::ModelHeader::headerSize)
                                + sizeof(HeaderLatest::ModelHeader::Version);
    if (stream_len > size_of_headers_header) {
        readNBytes(&header, size_of_headers_header, is);
    } else {
        readNBytes(&header, stream_len, is);
    }
    if (*reinterpret_cast<int*>(header.gnam) != gna_header_magic) {
        THROW_GNA_EXCEPTION << "Imported file unsupported: magic number should be GNAM(0x474e414d), but was 0x"
                           << std::setfill('0') <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[0]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[1]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[2]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[3]);
    }
    is.seekg(0, is.beg);
    Header2dot1::ModelHeader tempHeader2dot1;
    switch (header.version.major) {
        case 2:
            switch (header.version.minor) {
                case 1:
                    readBits(tempHeader2dot1, is);
                    header = Header2dot2::ModelHeader(tempHeader2dot1);
                    break;
                case 2:
                    readBits(header, is);
                    break;
                default:
                    THROW_GNA_EXCEPTION << "Imported file unsupported. minor version should be equal to 1 or 2 and is: " << header.version.minor;
    }
            break;
        default:
            THROW_GNA_EXCEPTION << "Imported file unsupported. Import for files with major version equal to: " << header.version.major << " is not implemented";
    }
    /*
     * extra data need to be added into new header and modify check as appropriate
     */

    //  forward compatible
    if (header.headerSize > sizeof(header)) {
        is.seekg(header.headerSize - sizeof(header), std::ios_base::cur);
    }
    return header;
}

GNAPluginNS::HeaderLatest::ModelHeader GNAModelSerial::ReadHeader(int fd) {
    auto stream_len = lseek(fd, 0, SEEK_END);
    if (stream_len == -1) {
        THROW_GNA_EXCEPTION << "Can't open file to import";
    }
    lseek(fd, 0, SEEK_SET);

    HeaderLatest::ModelHeader header;
    header.version.major = 0u;
    header.version.minor = 0u;
    auto size_of_headers_header = sizeof(HeaderLatest::ModelHeader::gnam) + sizeof(HeaderLatest::ModelHeader::headerSize)
                                + sizeof(HeaderLatest::ModelHeader::Version);
    if (stream_len > size_of_headers_header) {
        readNBytes(&header, size_of_headers_header, fd);
    } else {
        readNBytes(&header, stream_len, fd);
    }
    if (*reinterpret_cast<int*>(header.gnam) != gna_header_magic) {
        THROW_GNA_EXCEPTION << "Imported file unsupported: magic number should be GNAM(0x474e414d), but was 0x"
                           << std::setfill('0') <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[0]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[1]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[2]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[3]);
    }
    lseek(fd, 0, SEEK_SET);
    Header2dot1::ModelHeader tempHeader2dot1;
    switch (header.version.major) {
        case 2:
            switch (header.version.minor) {
                case 1:
                    readBits(tempHeader2dot1, fd);
                    header = Header2dot2::ModelHeader(tempHeader2dot1);
                    break;
                case 2:
                    readBits(header, fd);
                    break;
                default:
                    THROW_GNA_EXCEPTION << "Imported file unsupported. minor version should be equal to 1 or 2 and is: " << header.version.minor;
    }
            break;
        default:
            THROW_GNA_EXCEPTION << "Imported file unsupported. Import for files with major version equal to: " << header.version.major << " is not implemented";
    }
    /*
     * extra data need to be added into new header and modify check as appropriate
     */

    //  forward compatible
    if (header.headerSize > sizeof(header)) {
        lseek(fd, header.headerSize - sizeof(header), SEEK_CUR);
    }

    return header;
}

#define offsetFromBase(field)\
getOffsetFromBase(field, #field)

#if GNA_LIB_VER == 2

bool IsEmptyTensor(const Gna2Tensor& t) {
    return t.Type == Gna2DataTypeNone &&
        t.Data == nullptr &&
        t.Layout[0] == '\0' &&
        t.Mode == Gna2TensorModeDefault &&
        t.Shape.NumberOfDimensions == 0;
}

const std::map<Gna2OperationType, std::vector<uint32_t>> GnaParamSize{
    {Gna2OperationTypeFullyConnectedAffine, {sizeof(Gna2BiasMode), sizeof(uint32_t)}},
    {Gna2OperationTypeConvolution, {
        sizeof(Gna2Shape),
        sizeof(Gna2BiasMode),
        sizeof(Gna2PoolingMode),
        sizeof(Gna2Shape),
        sizeof(Gna2Shape),
        sizeof(Gna2Shape)}},
};

void GNAModelSerial::Import(void *basePointer,
        size_t gnaGraphSize,
        std::istream & is,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::InputsDataMap& inputsDataMap,
        InferenceEngine::OutputsDataMap& outputsDataMap) {
    is.exceptions(std::istream::failbit);
    for (auto inputIndex = 0; inputIndex < modelHeader.nInputs; inputIndex++) {
        uint32_t size = 0;
        readBits(size, is);
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, is);
        std::vector<char> inName(nameSize);
        readNBytes(inName.data(), nameSize, is);
        inputNames.emplace_back(std::string(inName.begin(), inName.end() - 1));
    }
    ImportInputs(is, basePointer, inputsDesc, inputsDataMap);
    for (auto inputIndex = 0; inputIndex < modelHeader.nOutputs; inputIndex++) {
        uint32_t size = 0;
        readBits(size, is);
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, is);
        std::vector<char> outName(nameSize);
        readNBytes(outName.data(), nameSize, is);
        outputNames.emplace_back(outName.begin(), outName.end() - 1);
    }
    ImportOutputs(is, basePointer, desc, outputsDataMap);

    for (auto operation = gna2Model->Operations; operation != gna2Model->Operations + gna2Model->NumberOfOperations; ++operation) {
        readNBits<32>(operation->Type, is);
        readBits(operation->NumberOfOperands, is);
        operation->Operands = static_cast<Gna2Tensor const **>(gnaUserAllocator(sizeof(Gna2Tensor*) * operation->NumberOfOperands));
        IE_ASSERT(operation->Operands != nullptr);
        for (uint32_t i = 0; i < operation->NumberOfOperands; i++) {
            Gna2Tensor t{};
            readBits(t, is);
            if (IsEmptyTensor(t)) {
                operation->Operands[i] = nullptr;
            } else {
                operation->Operands[i] = static_cast<Gna2Tensor const *>(gnaUserAllocator(sizeof(Gna2Tensor)));
                t.Data = offsetToPointer(basePointer, reinterpret_cast<uint64_t>(t.Data));
                const_cast<Gna2Tensor&>(*operation->Operands[i]) = t;
            }
        }
        readBits(operation->NumberOfParameters, is);
        switch (operation->Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Importing of recurrent operation not supported";
        case Gna2OperationTypeTransposition:
            THROW_GNA_EXCEPTION << "Importing of transposition operation not supported";
        case Gna2OperationTypeCopy:
            THROW_GNA_EXCEPTION << "Importing of copy operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA operation type(" << operation->Type << ")  not supported";
        }
        if (operation->NumberOfParameters > 0)
            operation->Parameters = static_cast<void **>(gnaUserAllocator(sizeof(void*) * operation->NumberOfParameters));
        else
            operation->Parameters = nullptr;
        for (uint32_t i = 0; i < operation->NumberOfParameters; i++) {
            uint32_t paramSize = 0;
            readBits(paramSize, is);
            IE_ASSERT(operation->Parameters != nullptr);
            if (paramSize == 0) {
                IE_ASSERT(operation->Parameters != nullptr);
                operation->Parameters[i] = nullptr;
                continue;
            }
            operation->Parameters[i] = gnaUserAllocator(paramSize);
            readNBytes(operation->Parameters[i], paramSize, is);

            if (GnaParamSize.at(operation->Type).size() <= i) {
                THROW_GNA_EXCEPTION << "Cannot import parameter of index: " << i;
            }
            if (paramSize != GnaParamSize.at(operation->Type).at(i)) {
                THROW_GNA_EXCEPTION << "Parameter size mismatch on import: " << i;
            }
        }
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, is);

    for (int i = 0; i != nStates; i++) {
         //Read the name back
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, is);
        std::vector<char> inName;
        for (uint32_t i = 0; i < nameSize; i++) {
            char next_char;
            readBits(next_char, is);
            inName.push_back(next_char);
        }
        auto memLayerName = std::string(inName.begin(), inName.end());

        float scaleFactor = 0;
        readNBits<64>(scaleFactor, is);

        void *pSegment;
        readOffset(pSegment, basePointer, is);
        uint32_t segmentSz;
        readBits(segmentSz, is);
        if (pstates) {
            (*pstates).emplace_back(memLayerName, pSegment, segmentSz, scaleFactor);
        }
    }


    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

void GNAModelSerial::Import(void *basePointer,
        size_t gnaGraphSize,
        int fd,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::InputsDataMap& inputsDataMap,
        InferenceEngine::OutputsDataMap& outputsDataMap) {
    for (auto inputIndex = 0; inputIndex < modelHeader.nInputs; inputIndex++) {
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, fd);
        std::vector<char> inName;
        for (uint32_t i = 0; i < nameSize; i++) {
            char next_char;
            readBits(next_char, fd);
            inName.push_back(next_char);
        }
        inputNames.emplace_back(std::string(inName.begin(), inName.end()));
    }
    ImportInputs(fd, basePointer, inputsDesc, inputsDataMap);
    for (auto inputIndex = 0; inputIndex < modelHeader.nOutputs; inputIndex++) {
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, fd);
        std::vector<char> outName;
        for (uint32_t i = 0; i < nameSize; i++) {
            char next_char;
            readBits(next_char, fd);
            outName.push_back(next_char);
        }
        outputNames.emplace_back(outName.begin(), outName.end());
    }
    ImportOutputs(fd, basePointer, desc, outputsDataMap);

    for (auto operation = gna2Model->Operations; operation != gna2Model->Operations + gna2Model->NumberOfOperations; ++operation) {
        readNBits<32>(operation->Type, fd);
        readBits(operation->NumberOfOperands, fd);
        operation->Operands = static_cast<Gna2Tensor const **>(gnaUserAllocator(sizeof(Gna2Tensor*) * operation->NumberOfOperands));
        IE_ASSERT(operation->Operands != nullptr);
        for (uint32_t i = 0; i < operation->NumberOfOperands; i++) {
            Gna2Tensor t{};
            readBits(t, fd);
            if (IsEmptyTensor(t)) {
                operation->Operands[i] = nullptr;
            } else {
                operation->Operands[i] = static_cast<Gna2Tensor const *>(gnaUserAllocator(sizeof(Gna2Tensor)));
                t.Data = offsetToPointer(basePointer, reinterpret_cast<uint64_t>(t.Data));
                const_cast<Gna2Tensor&>(*operation->Operands[i]) = t;
            }
        }
        readBits(operation->NumberOfParameters, fd);
        switch (operation->Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Importing of recurrent operation not supported";
        case Gna2OperationTypeTransposition:
            THROW_GNA_EXCEPTION << "Importing of transposition operation not supported";
        case Gna2OperationTypeCopy:
            THROW_GNA_EXCEPTION << "Importing of copy operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA operation type(" << operation->Type << ")  not supported";
        }
        if (operation->NumberOfParameters > 0)
            operation->Parameters = static_cast<void **>(gnaUserAllocator(sizeof(void*) * operation->NumberOfParameters));
        else
            operation->Parameters = nullptr;
        for (uint32_t i = 0; i < operation->NumberOfParameters; i++) {
            uint32_t paramSize = 0;
            readBits(paramSize, fd);
            IE_ASSERT(operation->Parameters != nullptr);
            if (paramSize == 0) {
                IE_ASSERT(operation->Parameters != nullptr);
                operation->Parameters[i] = nullptr;
                continue;
            }
            operation->Parameters[i] = gnaUserAllocator(paramSize);
            readNBytes(operation->Parameters[i], paramSize, fd);

            if (GnaParamSize.at(operation->Type).size() <= i) {
                THROW_GNA_EXCEPTION << "Cannot import parameter of index: " << i;
            }
            if (paramSize != GnaParamSize.at(operation->Type).at(i)) {
                THROW_GNA_EXCEPTION << "Parameter size mismatch on import: " << i;
            }
        }
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, fd);

    for (int i = 0; i != nStates; i++) {
         //Read the name back
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, fd);
        std::vector<char> inName;
        for (uint32_t i = 0; i < nameSize; i++) {
            char next_char;
            readBits(next_char, fd);
            inName.push_back(next_char);
        }
        auto memLayerName = std::string(inName.begin(), inName.end());

        float scaleFactor = 0;
        readBits(scaleFactor, fd);

        void *pSegment;
        readOffset(pSegment, basePointer, fd);
        uint32_t segmentSz;
        readBits(segmentSz, fd);
        if (pstates) {
            (*pstates).emplace_back(memLayerName, pSegment, segmentSz, scaleFactor);
        }
    }


    // once structure has been read lets read whole gna graph
    read(fd, reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

uint32_t guessGrouping(Gna2Model const& model) {
    if (model.NumberOfOperations == 0 ||
        model.Operations == nullptr ||
        model.Operations[0].Operands == nullptr ||
        model.Operations[0].NumberOfOperands == 0 ||
        model.Operations[0].Operands[0]->Shape.NumberOfDimensions < 2) {
        THROW_GNA_EXCEPTION << "Can not guess grouping";
    }
    return (std::min)(model.Operations[0].Operands[0]->Shape.Dimensions[0], model.Operations[0].Operands[0]->Shape.Dimensions[1]);
}

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, std::ostream & os) const {
    os.exceptions(std::ostream::failbit);

    const std::vector<Gna2Operation>
        layers(gna2Model->Operations, gna2Model->Operations + gna2Model->NumberOfOperations);


    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t>(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                << ") not in range segment retuned from GNAAlloc(0x" << basePointer << "-0x"
                << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto getTensorWithProperOffset = [&getOffsetFromBase](const Gna2Tensor& tensor) {
        Gna2Tensor out = tensor;
        out.Data = reinterpret_cast<void*>(getOffsetFromBase(tensor.Data));
        return out;
    };

    auto convert_to_serial = [getOffsetFromBase](const HeaderLatest::RuntimeEndPoint& ep) {
        HeaderLatest::RuntimeEndPoint out;
        out.elements_count = ep.elements_count;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
        out.scaleFactor = ep.scaleFactor;
        out.element_size = ep.element_size;
        out.orientation = ep.orientation;
        return out;
    };
    /**
     * writing header
     */
    HeaderLatest::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(HeaderLatest::ModelHeader);
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = guessGrouping(*gna2Model);
    header.nInputs = inputs.size();
    header.nOutputs = outputs.size();

    header.nRotateRows = nRotateRows;
    header.nRotateColumns = nRotateColumns;
    header.doRotateInput = doRotateInput;

    writeBits(header, os);

    for (auto &name : inputNames) {
        writeBits(name.size(), os);
        writeNBytes(name.c_str(), name.size(), os);
    }
    for (const auto &input : inputs) {
        writeBits(convert_to_serial(input), os);
    }
    for (auto &name : outputNames) {
        writeBits(name.size(), os);
        writeNBytes(name.c_str(), name.size(), os);
    }
    for (const auto &output : outputs) {
        writeBits(convert_to_serial(output), os);
    }
    for (const auto & layer : layers) {
        writeBits(static_cast<uint32_t>(layer.Type), os);
        writeBits(layer.NumberOfOperands, os);

        for (uint32_t i = 0; i < layer.NumberOfOperands; i++) {
            if (layer.Operands[i] == nullptr)
                writeBits(Gna2Tensor{}, os);
            else
                writeBits(getTensorWithProperOffset(*layer.Operands[i]), os);
        }

        writeBits(layer.NumberOfParameters, os);

        // writing parameters
        switch (layer.Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Exporting of recurrent operation not supported";
        case Gna2OperationTypeTransposition:
            THROW_GNA_EXCEPTION << "Exporting of interleave operation not supported";
        case Gna2OperationTypeCopy:
            THROW_GNA_EXCEPTION << "Exporting of copy operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Exporting of unknown GNA operation type(" << layer.Type << ")  not supported";
        }
        for (uint32_t i = 0; i < layer.NumberOfParameters; i++) {
            if (layer.Parameters[i] == nullptr) {
                writeBits(static_cast<uint32_t>(0), os);
                continue;
            }
            const auto paramSize = GnaParamSize.at(layer.Type).at(i);
            writeBits(paramSize, os);
            writeNBytes(layer.Parameters[i], paramSize, os);
        }
    }

    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), os);
    for (auto && state : states) {
        // State name
        writeBits(state.layerName.size(), os);
        writeNBytes(state.layerName.c_str(), state.layerName.size(), os);
        writeBits(state.scaleFactor, os);

        writeBits(offsetFromBase(state.ptr), os);
        writeBits(state.shape, os);
    }

    // once structure has been written lets push gna graph
    os.write(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, int fd) const {
    const std::vector<Gna2Operation>
        layers(gna2Model->Operations, gna2Model->Operations + gna2Model->NumberOfOperations);

    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t>(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                << ") not in range segment retuned from GNAAlloc(0x" << basePointer << "-0x"
                << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto getTensorWithProperOffset = [&getOffsetFromBase](const Gna2Tensor& tensor) {
        Gna2Tensor out = tensor;
        out.Data = reinterpret_cast<void*>(getOffsetFromBase(tensor.Data));
        return out;
    };

    auto convert_to_serial = [getOffsetFromBase](const HeaderLatest::RuntimeEndPoint& ep) {
        HeaderLatest::RuntimeEndPoint out;
        out.elements_count = ep.elements_count;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
        out.scaleFactor = ep.scaleFactor;
        out.element_size = ep.element_size;
        out.orientation = ep.orientation;
        return out;
    };
    /**
     * writing header
     */
    HeaderLatest::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(HeaderLatest::ModelHeader);
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = guessGrouping(*gna2Model);
    header.nInputs = inputs.size();
    header.nOutputs = outputs.size();

    header.nRotateRows = nRotateRows;
    header.nRotateColumns = nRotateColumns;
    header.doRotateInput = doRotateInput;

    writeBits(header, fd);

    for (auto &name : inputNames) {
        writeBits(name.size(), fd);
        writeNBytes(name.c_str(), name.size(), fd);
    }
    for (const auto &input : inputs) {
        writeBits(convert_to_serial(input), fd);
    }
    for (auto &name : outputNames) {
        writeBits(name.size(), fd);
        writeNBytes(name.c_str(), name.size(), fd);
    }
    for (const auto &output : outputs) {
        writeBits(convert_to_serial(output), fd);
    }
    for (const auto & layer : layers) {
        writeBits(static_cast<uint32_t>(layer.Type), fd);
        writeBits(layer.NumberOfOperands, fd);

        for (uint32_t i = 0; i < layer.NumberOfOperands; i++) {
            if (layer.Operands[i] == nullptr)
                writeBits(Gna2Tensor{}, fd);
            else
                writeBits(getTensorWithProperOffset(*layer.Operands[i]), fd);
        }

        writeBits(layer.NumberOfParameters, fd);

        // writing parameters
        switch (layer.Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Exporting of recurrent operation not supported";
        case Gna2OperationTypeTransposition:
            THROW_GNA_EXCEPTION << "Exporting of interleave operation not supported";
        case Gna2OperationTypeCopy:
            THROW_GNA_EXCEPTION << "Exporting of copy operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Exporting of unknown GNA operation type(" << layer.Type << ")  not supported";
        }
        for (uint32_t i = 0; i < layer.NumberOfParameters; i++) {
            if (layer.Parameters[i] == nullptr) {
                writeBits(static_cast<uint32_t>(0), fd);
                continue;
            }
            const auto paramSize = GnaParamSize.at(layer.Type).at(i);
            writeBits(paramSize, fd);
            writeNBytes(layer.Parameters[i], paramSize, fd);
        }
    }
    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), fd);
    for (auto && state : states) {
        // State name
        writeBits(state.layerName.size(), fd);
        writeNBytes(state.layerName.c_str(), state.layerName.size(), fd);
        writeBits(state.scaleFactor, fd);
        writeBits(offsetFromBase(state.ptr), fd);
        writeBits(state.shape, fd);
    }

    // once structure has been written lets push gna graph
    write(fd, reinterpret_cast<char*>(basePointer), gnaGraphSize);
}
#else

void GNAModelSerial::Import(void *basePointer,
        size_t gnaGraphSize,
        std::istream & is,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::InputsDataMap& inputsDataMap,
        InferenceEngine::OutputsDataMap& outputsDataMap) {
    is.exceptions(std::istream::failbit);
    ImportInputs(is, basePointer, inputsDesc, inputsDataMap);
    ImportOutputs(is, basePointer, desc, outputsDataMap);

    auto readPwl = [&is, basePointer](intel_pwl_func_t & value) {
        readBits(value.nSegments, is);
        if (value.nSegments != 0) {
            readOffset(value.pSegments, basePointer, is);
        } else {
            value.pSegments = nullptr;
        }
    };

    for (auto layer = ptr_nnet->pLayers; layer != ptr_nnet->pLayers + ptr_nnet->nLayers; ++layer) {
        readBits(layer->nInputColumns, is);
        readBits(layer->nInputRows, is);
        readBits(layer->nOutputColumns, is);
        readBits(layer->nOutputRows, is);
        readBits(layer->nBytesPerInput, is);
        readBits(layer->nBytesPerOutput, is);
        readBits(layer->nBytesPerIntermediateOutput, is);
        readNBits<32>(layer->nLayerKind, is);

        // reading layers structs
        switch (layer->nLayerKind) {
        case INTEL_AFFINE_DIAGONAL:
        case INTEL_AFFINE: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_affine_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_affine_layer_t structure.";
            }

            auto &affine = *reinterpret_cast<intel_affine_layer_t *>(layer->pLayerStruct);
            readBits(affine.affine.nBytesPerWeight, is);
            readBits(affine.affine.nBytesPerBias, is);
            readOffset(affine.affine.pWeights, basePointer, is);
            readOffset(affine.affine.pBiases, basePointer, is);
            readPwl(affine.pwl);
            break;
        }
        case INTEL_CONVOLUTIONAL: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_convolutional_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_convolutional_layer_t structure.";
            }

            auto &convolution = *reinterpret_cast<intel_convolutional_layer_t *>(layer->pLayerStruct);
            readBits(convolution.nFilterCoefficients, is);
            readBits(convolution.nBytesFilterCoefficient, is);
            readBits(convolution.nBytesBias, is);
            readBits(convolution.nFilters, is);
            readBits(convolution.nFeatureMaps, is);
            readBits(convolution.nFeatureMapRows, is);
            readBits(convolution.nFeatureMapColumns, is);
            readBits(convolution.nFilterRows, is);
            readOffset(convolution.pFilters, basePointer, is);
            readOffset(convolution.pBiases, basePointer, is);
            readBits(convolution.nPoolSize, is);
            readBits(convolution.nPoolStride, is);
            readBits(convolution.poolType, is);
            readPwl(convolution.pwl);
            break;
        }

        case INTEL_RECURRENT:
            THROW_GNA_EXCEPTION << "Importing of recurrent layer not supported";
        case INTEL_INTERLEAVE:
            THROW_GNA_EXCEPTION << "Importing of interleave layer not supported";
        case INTEL_DEINTERLEAVE:
            THROW_GNA_EXCEPTION << "Importing of deinterleave layer not supported";
        case INTEL_COPY:
            THROW_GNA_EXCEPTION << "Importing of copy layer not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA layer kind(" << layer->nLayerKind << ")  not supported";
        }

        // reading offsets of inputs/outputs
        readOffset(layer->pInputs, basePointer, is);
        readOffset(layer->pOutputsIntermediate, basePointer, is);
        readOffset(layer->pOutputs, basePointer, is);
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, is);
    if (pstates != nullptr) {
        pstates->resize(nStates);
    }

    for (int i = 0; i != nStates; i++) {
        //Read the name back
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, is);
        std::vector<char> inName;
        for (uint32_t i = 0; i < nameSize; i++) {
            char next_char;
            readBits(next_char, is);
            inName.push_back(next_char);
        }
        auto memLayerName = std::string(inName.begin(), inName.end());

        float scaleFactor = 1.0;
        readBits(scaleFactor, is);

        void *pSegment;
        readOffset(pSegment, basePointer, is);
        uint32_t segmentSz;
        readBits(segmentSz, is);
        if (pstates) {
            (*pstates).emplace_back(memLayerName, pSegment, segmentSz, scaleFactor);
        }
    }

    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

/**
 *
 * @param ptr_nnet
 * @param gnaAllocSize - it can be calculated based on nnet, however it will overcomplicate export
 * about base adress it is relatively easy to calculate
 * @param os
 */

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, std::ostream & os) const {
    os.exceptions(std::ostream::failbit);

    std::vector<intel_nnet_layer_t>
        layers(ptr_nnet->pLayers, ptr_nnet->pLayers + ptr_nnet->nLayers);


    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t >(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                               << ") not in range segment returned from GNAAlloc(0x" << basePointer << "-0x"
                               << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto writePwl = [&os, getOffsetFromBase] (intel_pwl_func_t & value) {
        writeBits(value.nSegments, os);
        // export require certain offset, since offset from base to nullptr cannot be correct, we are not store it at all
        if (value.nSegments != 0) {
            writeBits(offsetFromBase(value.pSegments), os);
        }
    };

    auto convert_to_serial = [getOffsetFromBase](const HeaderLatest::RuntimeEndPoint& ep){
        HeaderLatest::RuntimeEndPoint out;
        out.elements_count = ep.elements_count;
        out.element_size = ep.element_size;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
        out.scaleFactor = ep.scaleFactor;
        out.orientation = ep.orientation;
        return out;
    };
    /**
     * writing header
     */
    HeaderLatest::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.version.major = 1u;
    header.version.minor = 1u;
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = ptr_nnet->nGroup;
    header.nInputs = 1;
    header.nOutputs = 1;
    header.headerSize = sizeof(HeaderLatest::ModelHeader);
    header.nRotateRows = nRotateRows;
    header.nRotateColumns = nRotateColumns;


    writeBits(header, os);
    writeBits(convert_to_serial(inputs[0]), os);
    writeBits(convert_to_serial(outputs[0]), os);

    for (auto & layer : layers) {
        writeBits(layer.nInputColumns, os);
        writeBits(layer.nInputRows, os);
        writeBits(layer.nOutputColumns, os);
        writeBits(layer.nOutputRows, os);
        writeBits(layer.nBytesPerInput, os);
        writeBits(layer.nBytesPerOutput, os);
        writeBits(layer.nBytesPerIntermediateOutput, os);
        writeBits(static_cast<uint32_t>(layer.nLayerKind), os);

        // writing layers structs
        switch (layer.nLayerKind) {
            case INTEL_AFFINE_DIAGONAL:
            case INTEL_AFFINE: {
                auto &affine = *reinterpret_cast<intel_affine_layer_t *>(layer.pLayerStruct);
                writeBits(affine.affine.nBytesPerWeight, os);
                writeBits(affine.affine.nBytesPerBias, os);
                writeBits(offsetFromBase(affine.affine.pWeights), os);
                writeBits(offsetFromBase(affine.affine.pBiases), os);
                writePwl(affine.pwl);
                break;
            }
            case INTEL_CONVOLUTIONAL: {
                auto &convolution = *reinterpret_cast<intel_convolutional_layer_t *>(layer.pLayerStruct);
                writeBits(convolution.nFilterCoefficients, os);
                writeBits(convolution.nBytesFilterCoefficient, os);
                writeBits(convolution.nBytesBias, os);
                writeBits(convolution.nFilters, os);
                writeBits(convolution.nFeatureMaps, os);
                writeBits(convolution.nFeatureMapRows, os);
                writeBits(convolution.nFeatureMapColumns, os);
                writeBits(convolution.nFilterRows, os);
                writeBits(offsetFromBase(convolution.pFilters), os);
                writeBits(offsetFromBase(convolution.pBiases), os);
                writeBits(convolution.nPoolSize, os);
                writeBits(convolution.nPoolStride, os);
                writeBits(convolution.poolType, os);
                writePwl(convolution.pwl);
                break;
            }

            case INTEL_RECURRENT:
                THROW_GNA_EXCEPTION << "Exporting of recurrent layer not supported";
            case INTEL_INTERLEAVE:
                THROW_GNA_EXCEPTION << "Exporting of interleave layer not supported";
            case INTEL_DEINTERLEAVE:
                THROW_GNA_EXCEPTION << "Exporting of deinterleave layer not supported";
            case INTEL_COPY:
                THROW_GNA_EXCEPTION << "Exporting of copy layer not supported";
            default:
                THROW_GNA_EXCEPTION << "Exporting of unknown GNA layer kind(" << layer.nLayerKind << ")  not supported";
        }

        // writing offsets from base.
        writeBits(offsetFromBase(layer.pInputs), os);
        writeBits(offsetFromBase(layer.pOutputsIntermediate), os);
        writeBits(offsetFromBase(layer.pOutputs), os);
    }

    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), os);
    for (auto && state : states) {
        // State name
        writeBits(state.layerName.size(), os);
        writeNBytes(state.layerName.c_str(), state.layerName.size(), os);
        writeBits(state.scaleFactor, os);

        writeBits(offsetFromBase(state.ptr), os);
        writeBits(state.shape, os);
    }

    // once structure has been written lets push gna graph
    os.write(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

#endif
std::vector<HeaderLatest::RuntimeEndPoint> GNAModelSerial::serializeOutputs(const InferenceEngine::OutputsDataMap& outputsDataMap,
        const std::vector<GNAPluginNS::OutputDesc>& outputsDesc) {
    std::vector<HeaderLatest::RuntimeEndPoint> endPoints;
    std::size_t outputIndex = 0;
    for (auto const &output : outputsDataMap) {
        auto outputName = output.first;
        auto inputDims = output.second->getTensorDesc().getDims();
        uint32_t elementsCount = static_cast<uint32_t>(InferenceEngine::details::product(inputDims.begin(), inputDims.end()));

        HeaderLatest::RuntimeEndPoint endPoint(outputsDesc[outputIndex].scale_factor,
                                                 outputsDesc[outputIndex].ptrs[0],
                                                 outputsDesc[outputIndex].num_bytes_per_element,
                                                 elementsCount,
                                                 outputsDesc[outputIndex].orientation);
        endPoints.push_back(endPoint);
        outputIndex++;
    }
    return endPoints;
}

std::vector<HeaderLatest::RuntimeEndPoint> GNAModelSerial::serializeInputs(const InferenceEngine::InputsDataMap& inputsDataMap,
                                                                             std::shared_ptr<GNAPluginNS::InputDesc> inputDesc) {
    std::vector<HeaderLatest::RuntimeEndPoint> endPoints;

    std::size_t inputIndex = 0;
    for (auto const& input : inputsDataMap) {
        auto inputName = input.first;
        auto inputDims = input.second->getTensorDesc().getDims();

        double scaleFactor = inputDesc->getScaleFactor(inputIndex);
        std::vector<void *> descriptor_ptr = inputDesc->get_ptr_inputs_global(inputName);
        IE_ASSERT(descriptor_ptr.size() > 0);
        uint32_t element_size = 2u;
        uint32_t elementsCount = static_cast<uint32_t>(InferenceEngine::details::product(inputDims.begin(), inputDims.end()));
        intel_dnn_orientation_t orientation = inputDesc->getOrientation(inputName);

        HeaderLatest::RuntimeEndPoint endPoint(scaleFactor,
                                                 descriptor_ptr[0],
                                                 element_size,
                                                 elementsCount,
                                                 orientation);
        endPoints.push_back(endPoint);
        inputIndex++;
    }
    return endPoints;
}

void GNAModelSerial::ImportOutputs(int fd,
        void* basePtr,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::OutputsDataMap& dataMap) {
    desc.clear();
    dataMap.clear();
    desc.resize(modelHeader.nOutputs);

    for (auto outputIndex = 0; outputIndex < modelHeader.nOutputs; outputIndex++) {
        std::string name = outputNames.at(outputIndex);
        HeaderLatest::RuntimeEndPoint output;
        read(fd, reinterpret_cast<char *>(&output), sizeof(output));
        OutputDesc description;
        description.ptrs.push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + output.descriptor_offset));
        description.orientation = kDnnInterleavedOrientation;
        description.orientation = output.orientation;
        description.num_bytes_per_element = output.element_size;
        description.scale_factor = output.scaleFactor;

        auto outputDims = InferenceEngine::SizeVector({modelHeader.nGroup, output.elements_count / modelHeader.nGroup});
        dataMap[name] = std::make_shared<InferenceEngine::Data>(name,
                                                 InferenceEngine::TensorDesc(
                                                         InferenceEngine::Precision::FP32,
                                                         outputDims,
                                                         InferenceEngine::Layout::NC));
        desc.at(outputIndex) = description;
    }
}

void GNAModelSerial::ImportInputs(int fd,
        void* basePtr,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        InferenceEngine::InputsDataMap& dataMap) {
    dataMap.clear();

    for (auto inputIndex = 0; inputIndex < modelHeader.nInputs; inputIndex++) {
        std::string name = inputNames.at(inputIndex);
        HeaderLatest::RuntimeEndPoint input;
        read(fd, reinterpret_cast<char *>(&input), sizeof(input));
        inputsDesc->get_ptr_inputs_global(name).push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + input.descriptor_offset));
        inputsDesc->orientation_in[name] = input.orientation;
        inputsDesc->bytes_allocated_for_input[name] = input.element_size * input.elements_count;

        auto inputDims = InferenceEngine::SizeVector({modelHeader.nGroup, input.elements_count / modelHeader.nGroup});

        dataMap[name] = std::make_shared<InferenceEngine::InputInfo>();
        dataMap[name]->setInputData(std::make_shared<InferenceEngine::Data>(name,
                                                            InferenceEngine::TensorDesc(
                                                                    InferenceEngine::Precision::FP32,
                                                                    inputDims,
                                                                    InferenceEngine::Layout::NC)));
        inputsDesc->inputScaleFactors.push_back(input.scaleFactor);
    }
}

void GNAModelSerial::ImportInputs(std::istream &is,
        void* basePtr,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        InferenceEngine::InputsDataMap& dataMap) {
    dataMap.clear();

    for (auto inputIndex = 0; inputIndex < modelHeader.nInputs; inputIndex++) {
        std::string name = inputNames.at(inputIndex);
        HeaderLatest::RuntimeEndPoint input;
        is.read(reinterpret_cast<char *>(&input), sizeof(input));
        inputsDesc->get_ptr_inputs_global(name).push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + input.descriptor_offset));
        inputsDesc->orientation_in[name] = input.orientation;
        inputsDesc->bytes_allocated_for_input[name] = input.element_size * input.elements_count;

        auto inputDims = InferenceEngine::SizeVector({modelHeader.nGroup, input.elements_count / modelHeader.nGroup});

        dataMap[name] = std::make_shared<InferenceEngine::InputInfo>();
        dataMap[name]->setInputData(std::make_shared<InferenceEngine::Data>(name,
                                                            InferenceEngine::TensorDesc(
                                                                    InferenceEngine::Precision::FP32,
                                                                    inputDims,
                                                                    InferenceEngine::Layout::NC)));
        inputsDesc->inputScaleFactors.push_back(input.scaleFactor);
    }
}

void GNAModelSerial::ImportOutputs(std::istream &is,
        void* basePtr,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::OutputsDataMap& dataMap) {
    desc.clear();
    dataMap.clear();
    desc.resize(modelHeader.nOutputs);

    for (auto outputIndex = 0; outputIndex < modelHeader.nOutputs; outputIndex++) {
        std::string name = outputNames.at(outputIndex);
        HeaderLatest::RuntimeEndPoint output;
        is.read(reinterpret_cast<char *>(&output), sizeof(output));
        OutputDesc description;
        description.ptrs.push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + output.descriptor_offset));
        description.orientation = kDnnInterleavedOrientation;
        description.orientation = output.orientation;
        description.num_bytes_per_element = output.element_size;
        description.scale_factor = output.scaleFactor;

        auto outputDims = InferenceEngine::SizeVector({modelHeader.nGroup, output.elements_count / modelHeader.nGroup});
        dataMap[name] = std::make_shared<InferenceEngine::Data>(name,
                                                 InferenceEngine::TensorDesc(
                                                         InferenceEngine::Precision::FP32,
                                                         outputDims,
                                                         InferenceEngine::Layout::NC));
        desc.at(outputIndex) = description;
    }
}

void GNAModelSerial::setHeader(HeaderLatest::ModelHeader header) {
    modelHeader = header;
}
