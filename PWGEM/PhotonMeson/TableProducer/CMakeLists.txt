# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

add_subdirectory(converters)

o2physics_add_dpl_workflow(skimmer-gamma-conversion
                    SOURCES skimmerGammaConversion.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DCAFitter O2Physics::AnalysisCore KFParticle::KFParticle
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-gamma-conversion-truthonlymc
                    SOURCES skimmerGammaConversionTruthOnlyMc.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(photon-conversion-builder
                    SOURCES photonconversionbuilder.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DCAFitter O2Physics::AnalysisCore KFParticle::KFParticle
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(create-pcm
                    SOURCES createPCM.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DCAFitter O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(create-emevent
                    SOURCES createEMEvent.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(associate-mc-info
                    SOURCES associateMCinfo.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-gamma-calo
                    SOURCES skimmerGammaCalo.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-phos
                    SOURCES skimmerPHOS.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore O2::PHOSBase
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-primary-electron
                    SOURCES skimmerPrimaryElectron.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-primary-muon
                    SOURCES skimmerPrimaryMuon.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-dalitz-ee
                    SOURCES skimmerDalitzEE.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(skimmer-dalitz-mumu
                    SOURCES skimmerDalitzMuMu.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(gamma-table-producer
                    SOURCES gammaSelection.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)

o2physics_add_dpl_workflow(produce-meson-calo
                    SOURCES produceMesonCalo.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2Physics::AnalysisCore
                    COMPONENT_NAME Analysis)
