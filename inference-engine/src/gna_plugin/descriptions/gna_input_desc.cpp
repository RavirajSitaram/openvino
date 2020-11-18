// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "gna_input_desc.hpp"

std::vector<void *>& GNAPluginNS::InputDesc::get_ptr_inputs_global(std::string name) {
    if (!ptr_inputs_global_id.count(name)) {
        ptr_inputs_global_storage.push_front({});
        ptr_inputs_global_id[name] = ptr_inputs_global_storage.begin();
    }
    return *ptr_inputs_global_id[name];
}

intel_dnn_orientation_t GNAPluginNS::InputDesc::getOrientation(const std::string& name) {
    if (orientation_in.find(name) == orientation_in.end()) {
        THROW_GNA_EXCEPTION << "Can't find orientation for input name '" << name << "'";
    }
    return orientation_in[name];
}

float GNAPluginNS::InputDesc::getScaleFactor(const std::size_t index) {
    if (index >= inputScaleFactors.size()) {
        THROW_GNA_EXCEPTION << "Can't find scale factor for index = " << index;
    }
    return inputScaleFactors[index];
}
