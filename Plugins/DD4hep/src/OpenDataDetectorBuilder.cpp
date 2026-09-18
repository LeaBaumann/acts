// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/DD4hep/OpenDataDetectorBuilder.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/Blueprint.hpp"
#include "Acts/Geometry/BlueprintOptions.hpp"
#include "Acts/Geometry/ContainerBlueprintNode.hpp"
#include "Acts/Geometry/CylinderVolumeBounds.hpp"
#include "Acts/Geometry/Extent.hpp"
#include "Acts/Geometry/MaterialDesignatorBlueprintNode.hpp"
#include "Acts/Geometry/NavigationPolicyFactory.hpp"
#include "Acts/Geometry/VolumeAttachmentStrategy.hpp"
#include "Acts/Geometry/VolumeResizeStrategy.hpp"
#include "Acts/Navigation/CylinderNavigationPolicy.hpp"
#include "Acts/Navigation/SurfaceArrayNavigationPolicy.hpp"
#include "Acts/Utilities/AxisDefinitions.hpp"
#include "Acts/Utilities/AxisSpec.hpp"
#include "ActsPlugins/DD4hep/BlueprintBuilder.hpp"
#include "ActsPlugins/DD4hep/DD4hepConversionHelpers.hpp"

#include <format>
#include <initializer_list>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

#include <DD4hep/DetElement.h>
#include <DD4hep/Detector.h>

namespace ActsPlugins::DD4hep {

namespace {

// Looks up the first present material-binning parameter among the given
// keys (tried in order) and returns it. The ODD's DD4hep XML always sets the
// binning parameters together with the flag that gates them (see
// ODDHelper::xmlToProtoSurfaceMaterial in the ODD detector factories), so by
// the time this is called - always behind a check that the flag is set -
// one of the alternatives is guaranteed to be present. A missing value would
// mean the flag check and the XML have gone out of sync, which should fail
// loudly rather than silently guess a bin count.
std::size_t requireBinCount(const dd4hep::rec::VariantParameters& params,
                            std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (params.contains(key)) {
      return static_cast<std::size_t>(params.get<int>(key));
    }
  }
  throw std::runtime_error(
      "OpenDataDetectorBuilder: expected one of the given material binning "
      "parameters to be present, found none");
}

auto makeLayerCustomizer(const BlueprintBuilder& builder, std::string det,
                         std::regex layerFilter) {
  return [&builder, det = std::move(det), layerFilter = std::move(layerFilter)](
             const std::optional<dd4hep::DetElement>& elem,
             Acts::detail::LayerNodePtr layer) -> Acts::detail::BlueprintNodePtr {
    layer->setEnvelope(detail::kLayerEnvelope);

    const std::string elemName =
        elem.has_value() ? std::string{builder.backend().nameOf(*elem)}
                         : layer->name();
    const int layerIdx = detail::layerIndexFromName(elemName, layerFilter);

    using SrfArrayNavPol = Acts::SurfaceArrayNavigationPolicy;
    using enum SrfArrayNavPol::LayerType;

    SrfArrayNavPol::Config navCfg;
    navCfg.envelope = detail::kLayerEnvelope;

    const bool isCylinder =
        layer->layerType() == Acts::LayerBlueprintNode::LayerType::Cylinder;

    if (isCylinder) {
      // Barrel layer
      navCfg.layerType = Cylinder;
      navCfg.bins = {
          builder.backend().constant("{}_b{}_sf_b_phi", det, layerIdx),
          builder.backend().constant("{}_b_sf_b_z", det)};
    } else {
      // Endcap layer
      navCfg.layerType = Disc;
      navCfg.bins = {builder.backend().constant("{}_e_sf_b_r", det),
                     builder.backend().constant("{}_e_sf_b_phi", det)};
    }

    layer->setNavigationPolicyFactory(Acts::NavigationPolicyFactory{}
                                          .add<Acts::CylinderNavigationPolicy>()
                                          .add<SrfArrayNavPol>(navCfg)
                                          .asUniquePtr());

    // Only attach proto material where the ODD XML itself marks this layer -
    // the same "layer_material" flag Gen1's DD4hepLayerBuilder reads via
    // addCylinderLayerProtoMaterial/addDiscLayerProtoMaterial. Without this
    // gate every layer got material unconditionally, which is why Gen3's
    // marked surfaces didn't line up with Gen1's (e.g. the ODD XML puts
    // material on individual Pixel barrel layers, but on the *enclosing
    // container* for the Strip barrels, not on their individual layers).
    const bool hasLayerMaterial =
        elem.has_value() && getParamOr<bool>("layer_material", *elem, false);
    if (!hasLayerMaterial) {
      return layer;
    }

    const auto& params = getParams(*elem);

    using enum Acts::CylinderVolumeBounds::Face;
    auto matNode = std::make_shared<Acts::MaterialDesignatorBlueprintNode>(
        layer->name() + "_mat");

    if (isCylinder) {
      // Barrel: thin cylindrical shell -> outer mantle face carries material.
      // NOTE: do NOT also (or instead) configure InnerCylinder here - for any
      // layer that is not the innermost child of its radial
      // CylinderStackShell, that face gets fused with the neighbouring
      // sibling's face during Blueprint::construct and Acts throws
      // ("Material is designated on portal faces that are merged when
      // stacking child volumes"). This means we can't always honour the
      // exact face the ODD XML requests: Pixel barrel layers use
      // surface="outer" (fine, matches what we configure below), but Short/
      // LongStrip barrel layers use surface="inner" (would crash for every
      // layer but the innermost) - we approximate those with OuterCylinder
      // too rather than skip them. The bin counts still come from whichever
      // of the XML's outer/inner/representing variants is present, so the
      // material granularity matches Gen1 even though the face doesn't.
      const auto nPhi = requireBinCount(params, {"layer_material_outer_binPhi",
                                                 "layer_material_inner_binPhi",
                                                 "layer_material_representing_binPhi"});
      const auto nZ = requireBinCount(params, {"layer_material_outer_binZ",
                                               "layer_material_inner_binZ",
                                               "layer_material_representing_binZ"});
      matNode->configureFace(
          OuterCylinder,
          Acts::AxisSpec::DeferredEquidistant(nPhi, Acts::AxisDirection::AxisRPhi),
          Acts::AxisSpec::DeferredEquidistant(nZ, Acts::AxisDirection::AxisZ));
    } else {
      // Endcap: thin disc "pancake" -> both flat faces carry material. The
      // ODD marks endcap layers via surface="representing".
      const auto nR =
          requireBinCount(params, {"layer_material_representing_binR"});
      const auto nPhi =
          requireBinCount(params, {"layer_material_representing_binPhi"});
      matNode->configureFace(
          NegativeDisc,
          Acts::AxisSpec::DeferredEquidistant(nR, Acts::AxisDirection::AxisR),
          Acts::AxisSpec::DeferredEquidistant(nPhi, Acts::AxisDirection::AxisPhi));
      matNode->configureFace(
          PositiveDisc,
          Acts::AxisSpec::DeferredEquidistant(nR, Acts::AxisDirection::AxisR),
          Acts::AxisSpec::DeferredEquidistant(nPhi, Acts::AxisDirection::AxisPhi));
    }

    matNode->addChild(std::move(layer));
    return matNode;
  };
}

void addDirectLayerSubsystem(const BlueprintBuilder& builder,
                             Acts::ContainerBlueprintNode& outer,
                             std::string assembly, std::string det,
                             const std::regex& layerFilter) {
  const auto assemblyElement = builder.findDetElementByName(assembly);
  if (!assemblyElement.has_value()) {
    throw std::runtime_error(
        std::format("Could not find assembly '{}'", assembly));
  }

  auto barrels = builder.findBarrelElements(*assemblyElement);
  auto endcaps = builder.findEndcapElements(*assemblyElement);

  const std::string assemblyName{builder.backend().nameOf(*assemblyElement)};
  auto containerNode = std::make_shared<Acts::CylinderContainerBlueprintNode>(
      assemblyName, Acts::AxisDirection::AxisZ);

  auto layerCustomizer =
      makeLayerCustomizer(builder, std::move(det), layerFilter);

  auto addLayerChildren = [&](const auto& elements, auto makeNode) {
    for (const auto& element : elements) {
      auto node = makeNode(element);
      node->setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);
      node->setResizeStrategies(Acts::VolumeResizeStrategy::Gap,
                                Acts::VolumeResizeStrategy::Gap);
      containerNode->addChild(std::move(node));
    }
  };

  addLayerChildren(barrels, [&](const auto& barrel) {
    return builder.layers()
        .barrel()
        .setSensorAxes("XYZ")
        .setLayerFilter(layerFilter)
        .setContainer(barrel)
        .onLayer(layerCustomizer)
        .build();
  });

  addLayerChildren(endcaps, [&](const auto& endcap) {
    return builder.layers()
        .endcap()
        .setSensorAxes("XZY")
        .setLayerFilter(layerFilter)
        .setContainer(endcap)
        .onLayer(layerCustomizer)
        .build();
  });

  outer.addChild(std::move(containerNode));
}

void addBarrelEndcapSubsystem(const BlueprintBuilder& builder,
                              Acts::ContainerBlueprintNode& outer,
                              std::string assembly, std::string det,
                              const std::regex& layerFilter) {
  const auto assemblyElement = builder.findDetElementByName(assembly);
  if (!assemblyElement.has_value()) {
    throw std::runtime_error(
        std::format("Could not find assembly '{}'", assembly));
  }

  builder.barrelEndcap()
      .setAssembly(*assemblyElement)
      .setSensorAxes("XYZ", "XZY")
      .setLayerFilter(layerFilter)
      .onLayer(makeLayerCustomizer(builder, std::move(det), layerFilter))
      .onContainer([](const dd4hep::DetElement& elem,
                      Acts::detail::ContainerNodePtr node)
                       -> Acts::detail::BlueprintNodePtr {
        node->setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);
        node->setResizeStrategies(Acts::VolumeResizeStrategy::Gap,
                                  Acts::VolumeResizeStrategy::Gap);

        // The ODD XML sets "boundary_material" on the *barrel* container
        // (e.g. PixelBarrel), not on the individual layers, with up to three
        // faces: negative/positive (the barrel's own z-ends) and outer (its
        // mantle). This callback also runs for the endcap containers, where
        // the flag is simply absent, so nothing is wrapped there.
        //
        // NOTE: we only honour negative/positive here, not outer. The
        // barrel container's OuterCylinder face is, by construction, at
        // exactly the same radius as its outermost child layer's own
        // OuterCylinder - the two designations target the identical fused
        // portal and Acts throws ("Material is designated on portal faces
        // that are merged when stacking child volumes"). Per-layer material
        // (see makeLayerCustomizer) already covers that radius, so the
        // container-level "outer" entry is dropped rather than the
        // per-layer one.
        if (!getParamOr<bool>("boundary_material", elem, false) ||
            (!getParamOr<bool>("boundary_material_negative", elem, false) &&
             !getParamOr<bool>("boundary_material_positive", elem, false))) {
          return node;
        }

        const auto& params = getParams(elem);

        using enum Acts::CylinderVolumeBounds::Face;
        auto matNode = std::make_shared<Acts::MaterialDesignatorBlueprintNode>(
            node->name() + "_bmat");

        if (getParamOr<bool>("boundary_material_negative", elem, false)) {
          const auto nR = requireBinCount(params,
                                          {"boundary_material_negative_binR"});
          const auto nPhi = requireBinCount(
              params, {"boundary_material_negative_binPhi"});
          matNode->configureFace(
              NegativeDisc,
              Acts::AxisSpec::DeferredEquidistant(nR, Acts::AxisDirection::AxisR),
              Acts::AxisSpec::DeferredEquidistant(nPhi,
                                                  Acts::AxisDirection::AxisPhi));
        }
        if (getParamOr<bool>("boundary_material_positive", elem, false)) {
          const auto nR = requireBinCount(params,
                                          {"boundary_material_positive_binR"});
          const auto nPhi = requireBinCount(
              params, {"boundary_material_positive_binPhi"});
          matNode->configureFace(
              PositiveDisc,
              Acts::AxisSpec::DeferredEquidistant(nR, Acts::AxisDirection::AxisR),
              Acts::AxisSpec::DeferredEquidistant(nPhi,
                                                  Acts::AxisDirection::AxisPhi));
        }
        matNode->addChild(node);
        return matNode;
      })
      .addTo(outer);
}

void addDirectLayerGroupedSubsystem(const BlueprintBuilder& builder,
                                    Acts::ContainerBlueprintNode& outer,
                                    std::string assembly, std::string det,
                                    const std::regex& layerFilter) {
  const auto assemblyElement = builder.findDetElementByName(assembly);
  if (!assemblyElement.has_value()) {
    throw std::runtime_error(
        std::format("Could not find assembly '{}'", assembly));
  }

  auto barrels = builder.findBarrelElements(*assemblyElement);
  auto endcaps = builder.findEndcapElements(*assemblyElement);

  const std::string assemblyName{builder.backend().nameOf(*assemblyElement)};
  auto containerNode = std::make_shared<Acts::CylinderContainerBlueprintNode>(
      assemblyName, Acts::AxisDirection::AxisZ);

  auto layerCustomizer =
      makeLayerCustomizer(builder, std::move(det), layerFilter);

  auto sensorToLayerKey = [&](const dd4hep::DetElement& elem) {
    auto current = elem;
    const auto world = builder.backend().world();
    while (!(current == world)) {
      std::cmatch match;
      if (const std::string name{builder.backend().nameOf(current)};
          std::regex_search(name.c_str(), match, layerFilter) &&
          match.size() > 1) {
        return builder.getPathToElementName(current);
      }
      current = builder.backend().parent(current);
    }
    return builder.getPathToElementName(elem);
  };

  for (const auto& barrel : barrels) {
    auto sensors = builder.resolveSensitives(barrel);
    auto barrelNode = builder.layersFromSensors()
                          .barrel()
                          .setSensorAxes("XYZ")
                          .setSensors(std::move(sensors))
                          .setContainerName(builder.backend().nameOf(barrel))
                          .groupBy(sensorToLayerKey)
                          .onLayer(layerCustomizer)
                          .build();
    barrelNode->setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);
    barrelNode->setResizeStrategies(Acts::VolumeResizeStrategy::Gap,
                                    Acts::VolumeResizeStrategy::Gap);
    containerNode->addChild(std::move(barrelNode));
  }

  for (const auto& endcap : endcaps) {
    auto sensors = builder.resolveSensitives(endcap);
    auto endcapNode = builder.layersFromSensors()
                          .endcap()
                          .setSensorAxes("XZY")
                          .setSensors(std::move(sensors))
                          .setContainerName(builder.backend().nameOf(endcap))
                          .groupBy(sensorToLayerKey)
                          .onLayer(layerCustomizer)
                          .build();
    endcapNode->setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);
    endcapNode->setResizeStrategies(Acts::VolumeResizeStrategy::Gap,
                                    Acts::VolumeResizeStrategy::Gap);
    containerNode->addChild(std::move(endcapNode));
  }

  outer.addChild(std::move(containerNode));
}

}  // namespace

std::unique_ptr<Acts::TrackingGeometry> buildOpenDataDetectorBarrelEndcap(
    const dd4hep::Detector& detector, const Acts::GeometryContext& gctx,
    const Acts::Logger& logger) {
  using namespace Acts;
  using enum AxisDirection;

  BlueprintBuilder builder{{
                               .dd4hepDetector = &detector,
                               .lengthScale = Acts::UnitConstants::cm,
                               .gctx = gctx,
                           },
                           logger.cloneWithSuffix("BlpBld")};

  Blueprint::Config blueprintCfg;
  blueprintCfg.envelope = ActsPlugins::DD4hep::detail::kBlueprintEnvelope;
  Blueprint root{blueprintCfg};

  auto& outer = root.addCylinderContainer("OpenDataDetector", AxisR);
  outer.setAttachmentStrategy(VolumeAttachmentStrategy::Gap);

  const auto beampipeElement = builder.findDetElementByName("BeamPipe");
  if (!beampipeElement.has_value()) {
    throw std::runtime_error("Could not find BeamPipe element");
  }
  outer.addMaterial("Beampipe_mat", [&](Acts::MaterialDesignatorBlueprintNode& mat) {
    using enum Acts::CylinderVolumeBounds::Face;
    const auto& params = getParams(*beampipeElement);
    const auto nPhi =
        requireBinCount(params, {"layer_material_representing_binPhi"});
    const auto nZ =
        requireBinCount(params, {"layer_material_representing_binZ"});
    mat.configureFace(OuterCylinder,
                      Acts::AxisSpec::DeferredEquidistant(nPhi, AxisRPhi),
                      Acts::AxisSpec::DeferredEquidistant(nZ, AxisZ));
    mat.addChild(builder.backend().makeBeampipe());
  });

  addBarrelEndcapSubsystem(builder, outer, "Pixels", "pix",
                           ActsPlugins::DD4hep::detail::kPixelLayerFilter);

  // The passive support tube (PST) sits radially between the Pixel and
  // ShortStrip systems (pst_rmin/rmax ~ 202-204mm, vs. ~180mm Pixel outer
  // layer and 240mm ShortStrip inner layer in the ODD). It is not part of
  // any barrel-endcap assembly, so it needs to be added to the outer radial
  // stack directly, like the beampipe. Unlike the beampipe, its absence is
  // tolerated (older/other ODD variants may not have one).
  if (const auto pstElement = builder.findDetElementByName("PST");
      pstElement.has_value()) {
    outer.addMaterial("PST_mat", [&](Acts::MaterialDesignatorBlueprintNode& mat) {
      using enum Acts::CylinderVolumeBounds::Face;
      const auto& params = getParams(*pstElement);
      const auto nPhi =
          requireBinCount(params, {"layer_material_representing_binPhi"});
      const auto nZ =
          requireBinCount(params, {"layer_material_representing_binZ"});
      mat.configureFace(OuterCylinder,
                        Acts::AxisSpec::DeferredEquidistant(nPhi, AxisRPhi),
                        Acts::AxisSpec::DeferredEquidistant(nZ, AxisZ));
      mat.addChild(builder.backend().makePassiveCylinder(*pstElement));
    });
  }

  addBarrelEndcapSubsystem(builder, outer, "ShortStrips", "ss",
                           ActsPlugins::DD4hep::detail::kShortStripLayerFilter);
  addBarrelEndcapSubsystem(builder, outer, "LongStrips", "ls",
                           ActsPlugins::DD4hep::detail::kLongStripLayerFilter);

  return root.construct(BlueprintOptions{}, gctx, logger);
}

std::unique_ptr<Acts::TrackingGeometry> buildOpenDataDetectorDirectLayer(
    const dd4hep::Detector& detector, const Acts::GeometryContext& gctx,
    const Acts::Logger& logger) {
  using namespace Acts;
  using enum AxisDirection;

  BlueprintBuilder builder{{
                               .dd4hepDetector = &detector,
                               .lengthScale = Acts::UnitConstants::cm,
                               .gctx = gctx,
                           },
                           logger.cloneWithSuffix("BlpBld")};

  Blueprint::Config blueprintCfg;
  blueprintCfg.envelope = ActsPlugins::DD4hep::detail::kBlueprintEnvelope;
  Blueprint root{blueprintCfg};

  auto& outer = root.addCylinderContainer("OpenDataDetector", AxisR);
  outer.setAttachmentStrategy(VolumeAttachmentStrategy::Gap);

  const auto beampipeElement = builder.findDetElementByName("BeamPipe");
  if (!beampipeElement.has_value()) {
    throw std::runtime_error("Could not find BeamPipe element");
  }
  outer.addMaterial("Beampipe_mat", [&](Acts::MaterialDesignatorBlueprintNode& mat) {
    using enum Acts::CylinderVolumeBounds::Face;
    const auto& params = getParams(*beampipeElement);
    const auto nPhi =
        requireBinCount(params, {"layer_material_representing_binPhi"});
    const auto nZ =
        requireBinCount(params, {"layer_material_representing_binZ"});
    mat.configureFace(OuterCylinder,
                      Acts::AxisSpec::DeferredEquidistant(nPhi, AxisRPhi),
                      Acts::AxisSpec::DeferredEquidistant(nZ, AxisZ));
    mat.addChild(builder.backend().makeBeampipe());
  });

  addDirectLayerSubsystem(builder, outer, "Pixels", "pix",
                          ActsPlugins::DD4hep::detail::kPixelLayerFilter);
  addDirectLayerSubsystem(builder, outer, "ShortStrips", "ss",
                          ActsPlugins::DD4hep::detail::kShortStripLayerFilter);
  addDirectLayerSubsystem(builder, outer, "LongStrips", "ls",
                          ActsPlugins::DD4hep::detail::kLongStripLayerFilter);

  return root.construct(BlueprintOptions{}, gctx, logger);
}

std::unique_ptr<Acts::TrackingGeometry> buildOpenDataDetectorDirectLayerGrouped(
    const dd4hep::Detector& detector, const Acts::GeometryContext& gctx,
    const Acts::Logger& logger) {
  using namespace Acts;
  using enum AxisDirection;

  BlueprintBuilder builder{{
                               .dd4hepDetector = &detector,
                               .lengthScale = Acts::UnitConstants::cm,
                               .gctx = gctx,
                           },
                           logger.cloneWithSuffix("BlpBld")};

  Blueprint::Config blueprintCfg;
  blueprintCfg.envelope = ActsPlugins::DD4hep::detail::kBlueprintEnvelope;
  Blueprint root{blueprintCfg};

  auto& outer = root.addCylinderContainer("OpenDataDetector", AxisR);
  outer.setAttachmentStrategy(VolumeAttachmentStrategy::Gap);

  const auto beampipeElement = builder.findDetElementByName("BeamPipe");
  if (!beampipeElement.has_value()) {
    throw std::runtime_error("Could not find BeamPipe element");
  }
  outer.addMaterial("Beampipe_mat", [&](Acts::MaterialDesignatorBlueprintNode& mat) {
    using enum Acts::CylinderVolumeBounds::Face;
    const auto& params = getParams(*beampipeElement);
    const auto nPhi =
        requireBinCount(params, {"layer_material_representing_binPhi"});
    const auto nZ =
        requireBinCount(params, {"layer_material_representing_binZ"});
    mat.configureFace(OuterCylinder,
                      Acts::AxisSpec::DeferredEquidistant(nPhi, AxisRPhi),
                      Acts::AxisSpec::DeferredEquidistant(nZ, AxisZ));
    mat.addChild(builder.backend().makeBeampipe());
  });

  addDirectLayerGroupedSubsystem(
      builder, outer, "Pixels", "pix",
      ActsPlugins::DD4hep::detail::kPixelLayerFilter);
  addDirectLayerGroupedSubsystem(
      builder, outer, "ShortStrips", "ss",
      ActsPlugins::DD4hep::detail::kShortStripLayerFilter);
  addDirectLayerGroupedSubsystem(
      builder, outer, "LongStrips", "ls",
      ActsPlugins::DD4hep::detail::kLongStripLayerFilter);

  return root.construct(BlueprintOptions{}, gctx, logger);
}

}  // namespace ActsPlugins::DD4hep
