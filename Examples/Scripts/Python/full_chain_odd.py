#!/usr/bin/env python3

import os
import argparse
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uproot
import awkward as ak
import numpy as np

import acts
import acts.examples
from acts.examples.simulation import (
    MomentumConfig,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    ParticleSelectorConfig,
    addParticleGun,
    addPythia8,
    addGenParticleSelection,
    addFatras,
    addGeant4,
    addSimParticleSelection,
    addDigitization,
    addDigiParticleSelection,
)
from acts.examples.reconstruction import (
    addSeeding,
    CkfConfig,
    addCKFTracks,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addAmbiguityResolutionML,
    AmbiguityResolutionMLConfig,
    addScoreBasedAmbiguityResolution,
    ScoreBasedAmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder,
    addSeedFilterML,
    SeedFilterMLDBScanConfig,
)
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

u = acts.UnitConstants


parser = argparse.ArgumentParser(description="Full chain with the OpenDataDetector")
parser.add_argument(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "odd_output",
)
parser.add_argument("--events", "-n", help="Number of events", type=int, default=100)
parser.add_argument("--skip", "-s", help="Number of events", type=int, default=0)
parser.add_argument(
    "--jobs",
    "-j",
    help="Number of worker threads (-1 uses all cores)",
    type=int,
    default=None,
)
parser.add_argument("--edm4hep", help="Use edm4hep inputs", type=pathlib.Path)
parser.add_argument(
    "--geant4", help="Use Geant4 instead of fatras", action="store_true"
)
parser.add_argument(
    "--ttbar",
    help="Use Pythia8 (ttbar, pile-up 200) instead of particle gun",
    action="store_true",
)
parser.add_argument(
    "--ttbar-pu",
    help="Number of pile-up events for ttbar",
    type=int,
    default=200,
)
parser.add_argument(
    "--gun-particles",
    help="Multiplicity (no. of particles) of the particle gun",
    type=int,
    default=4,
)
parser.add_argument(
    "--gun-multiplicity",
    help="Multiplicity (no. of vertices) of the particle gun",
    type=int,
    default=200,
)
parser.add_argument(
    "--gun-eta-range",
    nargs=2,
    help="Eta range of the particle gun",
    type=float,
    default=[-3.0, 3.0],
)
parser.add_argument(
    "--gun-pt-range",
    nargs=2,
    help="Pt range of the particle gun (GeV)",
    type=float,
    default=[1.0 * u.GeV, 10.0 * u.GeV],
)
parser.add_argument(
    "--digi-config", help="Digitization configuration file", type=pathlib.Path
)
parser.add_argument(
    "--material-config", help="Material map configuration file", type=pathlib.Path
)
parser.add_argument(
    "--ambi-solver",
    help="Set which ambiguity solver to use, default is the classical one",
    type=str,
    choices=["greedy", "scoring", "ML"],
    default="greedy",
)
parser.add_argument(
    "--ambi-config",
    help="Set the configuration file for the Score Based ambiguity resolution",
    type=pathlib.Path,
    default=pathlib.Path.cwd() / "ambi_config.json",
)

parser.add_argument(
    "--MLSeedFilter",
    help="Use the Ml seed filter to select seed after the seeding step",
    action="store_true",
)
parser.add_argument(
    "--reco",
    help="Switch reco on/off",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--output-root",
    help="Switch root output on/off",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--output-csv",
    help="Switch csv output on/off",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--output-obj",
    help="Switch obj output on/off",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--output-parquet",
    help="Switch parquet output on/off (requires ACTS_BUILD_EXAMPLES_PARQUET=ON)",
    default=False,
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()

outputDir = args.output
ambi_ML = args.ambi_solver == "ML"
ambi_scoring = args.ambi_solver == "scoring"
ambi_config = args.ambi_config
seedFilter_ML = args.MLSeedFilter
geoDir = getOpenDataDetectorDirectory()
actsDir = pathlib.Path(__file__).parent.parent.parent.parent
# acts.examples.dump_args_calls()  # show python binding calls

oddMaterialMap = (
    args.material_config
    if args.material_config
    else geoDir / "data/odd-material-maps.root"
)

oddDigiConfig = (
    args.digi_config
    if args.digi_config
    else actsDir / "Examples/Configs/odd-digi-smearing-config.json"
)

oddSeedingSel = actsDir / "Examples/Configs/odd-seeding-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

generation = "gen3"
if generation == "gen1":
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
    print("gen1")
elif generation == "gen3":
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco, gen3=True)
    print("gen3")
else:
    print("no valid generation chosen")

trackingGeometry = detector.trackingGeometry()
decorators = detector.contextDecorators()
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args.events,
    skip=args.skip,
    numThreads=args.jobs if args.jobs is not None else (1 if args.geant4 else -1),
    outputDir=str(outputDir),
)

if args.edm4hep:
    import acts.examples.edm4hep
    from acts.examples.edm4hep import PodioReader

    s.addReader(
        PodioReader(
            level=acts.logging.DEBUG,
            inputPath=str(args.edm4hep),
            outputFrame="events",
            category="events",
        )
    )

    edm4hepReader = acts.examples.edm4hep.EDM4hepSimInputConverter(
        inputFrame="events",
        inputSimHits=[
            "PixelBarrelReadout",
            "PixelEndcapReadout",
            "ShortStripBarrelReadout",
            "ShortStripEndcapReadout",
            "LongStripBarrelReadout",
            "LongStripEndcapReadout",
        ],
        outputParticlesGenerator="particles_generated",
        outputParticlesSimulation="particles_simulated",
        outputSimHits="simhits",
        outputSimVertices="vertices_truth",
        dd4hepDetector=detector,
        trackingGeometry=trackingGeometry,
        sortSimHitsInTime=False,
        particleRMax=1080 * u.mm,
        particleZ=(-3030 * u.mm, 3030 * u.mm),
        particlePtMin=150 * u.MeV,
        level=acts.logging.DEBUG,
    )
    s.addAlgorithm(edm4hepReader)

    s.addWhiteboardAlias("particles", edm4hepReader.config.outputParticlesSimulation)

    addSimParticleSelection(
        s,
        ParticleSelectorConfig(
            rho=(0.0, 24 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-3.0, 3.0),
            removeNeutral=True,
        ),
    )
else:
    if not args.ttbar:
        addParticleGun(
            s,
            MomentumConfig(
                args.gun_pt_range[0] * u.GeV,
                args.gun_pt_range[1] * u.GeV,
                transverse=True,
            ),
            EtaConfig(args.gun_eta_range[0], args.gun_eta_range[1]),
            PhiConfig(0.0, 360.0 * u.degree),
            ParticleConfig(
                args.gun_particles, acts.PdgParticle.eMuon, randomizeCharge=True
            ),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(
                    0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 1.0 * u.ns
                ),
            ),
            multiplicity=args.gun_multiplicity,
            rnd=rnd,
        )
    else:
        addPythia8(
            s,
            hardProcess=["Top:qqbar2ttbar=on"],
            npileup=args.ttbar_pu,
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(
                    0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns
                ),
            ),
            rnd=rnd,
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
        )

        addGenParticleSelection(
            s,
            ParticleSelectorConfig(
                rho=(0.0, 24 * u.mm),
                absZ=(0.0, 1.0 * u.m),
                eta=(-3.0, 3.0),
                pt=(150 * u.MeV, None),
            ),
        )

    if args.geant4:
        if s.config.numThreads != 1:
            raise ValueError("Geant 4 simulation does not support multi-threading")

        # Pythia can sometime simulate particles outside the world volume, a cut on the Z of the track help mitigate this effect
        # Older version of G4 might not work, this as has been tested on version `geant4-11-00-patch-03`
        # For more detail see issue #1578
        addGeant4(
            s,
            detector,
            trackingGeometry,
            field,
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
            outputDirObj=outputDir if args.output_obj else None,
            rnd=rnd,
            killVolume=trackingGeometry.highestTrackingVolume,
            killAfterTime=25 * u.ns,
        )
    else:
        addFatras(
            s,
            trackingGeometry,
            field,
            enableInteractions=True,
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
            outputDirObj=outputDir if args.output_obj else None,
            rnd=rnd,
        )

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=oddDigiConfig,
    outputDirRoot=outputDir if args.output_root else None,
    outputDirCsv=outputDir if args.output_csv else None,
    rnd=rnd,
)

addDigiParticleSelection(
    s,
    ParticleSelectorConfig(
        pt=(1.0 * u.GeV, None),
        eta=(-3.0, 3.0),
        measurements=(9, None),
        removeNeutral=True,
    ),
)


class LeasVisitor(acts.TrackingGeometryMutableVisitor):
    """Visitor collecting cylinder volume rectangles and marker line segments."""
    def __init__(self, gctx: acts.GeometryContext):
        super().__init__()
        self._gctx = gctx
        self.volumes = [] # (z_min, z_max, r_min, r_max, name, vol_id)

    def visitVolume(self, volume: acts.Volume):
        bounds = volume.volumeBounds
        if bounds.type() != acts.VolumeBoundsType.Cylinder:
            return
        values = bounds.values()
        r_min, r_max, half_z = values[0], values[1], values[2]
        z_center = volume.center(self._gctx)[2]
        name = volume.volumeName if isinstance(volume, acts.TrackingVolume) else ""
        self.volumes.append((z_center - half_z, z_center + half_z, r_min, r_max, name, volume.geometryId.volume))


class LeasLayerVisitor(acts.TrackingGeometryMutableVisitor):
    def __init__(self, gctx: acts.GeometryContext):
        super().__init__()
        self._gctx = gctx
        self.layers = [] # (z_min, z_max, r_min, r_max, vol_id, lay_id)
        self.num_layers = 0

    def visitLayer(self, layer: acts.Layer):
        self.num_layers += 1
        surface = layer.surfaceRepresentation
        geo_id = surface.geometryId
        vol_id, lay_id = geo_id.volume, geo_id.layer
        bounds = surface.bounds
        values = bounds.values()
        z_center = surface.center(self._gctx)[2]
        thickness = 2.0

        if isinstance(surface, acts.CylinderSurface):
            r = bounds.get(acts.CylinderBoundsValue.R)
            r_min = r - (thickness / 2.0)
            r_max = r + (thickness / 2.0)
            half_z = bounds.get(acts.CylinderBoundsValue.HalfLengthZ)
            self.layers.append((z_center - half_z, z_center + half_z, r_min, r_max, vol_id, lay_id))
        elif isinstance(surface, acts.DiscSurface):
            if isinstance(bounds, acts.RadialBounds):
                r_min = values[acts.RadialBoundsValue.MinR]
                r_max = values[acts.RadialBoundsValue.MaxR]
            elif isinstance(bounds, acts.AnnulusBounds):
                r_min = values[acts.AnnulusBoundsValue.MinR]
                r_max = values[acts.AnnulusBoundsValue.MaxR]
            z_min = z_center - (thickness / 2.0)
            z_max = z_center + (thickness / 2.0)
            self.layers.append((z_min, z_max, r_min, r_max, vol_id, lay_id))


class LeasSensitiveVisitor(acts.TrackingGeometryMutableVisitor):
    def __init__(self, gctx: acts.GeometryContext):
        super().__init__()
        self._gctx = gctx
        self.sensitives = []  # (z_min, z_max, r_min, r_max, vol_id, lay_id, sens_id)
        self.n_total_calls = 0
        self.n_sensitive = 0
        self.n_skipped = 0

    def visitSurface(self, surface: acts.Surface):
        self.n_total_calls += 1
        geo_id = surface.geometryId
        sens_id = geo_id.sensitive
        if sens_id == 0:
            return
        self.n_sensitive += 1
        vol_id, lay_id = geo_id.volume, geo_id.layer

        bounds = surface.bounds
        values = bounds.values()

        if isinstance(bounds, acts.RectangleBounds):
            half_x, half_y = values[0], values[1]
            local_corners = [
                (-half_x, -half_y), (half_x, -half_y),
                (half_x, half_y), (-half_x, half_y),
            ]
        elif isinstance(bounds, acts.TrapezoidBounds):
            half_x_neg_y, half_x_pos_y, half_y = values[0], values[1], values[2]
            local_corners = [
                (-half_x_neg_y, -half_y), (half_x_neg_y, -half_y),
                (half_x_pos_y, half_y), (-half_x_pos_y, half_y),
            ]
        else:
            self.n_skipped += 1
            return

        direction = surface.normal(
            self._gctx,
            surface.center(self._gctx),
            acts.Vector3(0, 0, 1),  # dummy direction, irrelevant for a flat PlaneSurface
        )

        z_vals = []
        r_vals = []
        for lx, ly in local_corners:
            local_pos = acts.Vector2(lx, ly)
            global_pos = surface.localToGlobal(self._gctx, local_pos, direction)
            z_vals.append(global_pos[2])
            r_vals.append(np.sqrt(global_pos[0] ** 2 + global_pos[1] ** 2))

        z_min, z_max = min(z_vals), max(z_vals)
        r_min, r_max = min(r_vals), max(r_vals)

        self.sensitives.append((z_min, z_max, r_min, r_max, vol_id, lay_id, sens_id))


def build_material_rects(surfaces, gctx, viz_thickness=2.0):
    """Convert a list of material surfaces into rectangles for draw_rectangles().

    Returns tuples (z_min, z_max, r_min, r_max, vol_id, lay_id). Cylinder and disc
    representing surfaces are drawn with an artificial thickness so thin surfaces
    stay visible; planar surfaces are projected from their local corners, mirroring
    the layer/sensitive visitors above.
    """
    rects = []
    for surface in surfaces:
        geo_id = surface.geometryId
        vol_id, lay_id = geo_id.volume, geo_id.layer
        bounds = surface.bounds
        values = bounds.values()
        z_center = surface.center(gctx)[2]

        if isinstance(bounds, acts.CylinderBounds):
            r = bounds.get(acts.CylinderBoundsValue.R)
            half_z = bounds.get(acts.CylinderBoundsValue.HalfLengthZ)
            rects.append((z_center - half_z, z_center + half_z,
                          r - viz_thickness / 2.0, r + viz_thickness / 2.0,
                          vol_id, lay_id))

        elif isinstance(bounds, acts.RadialBounds):
            rects.append((z_center - viz_thickness / 2.0, z_center + viz_thickness / 2.0,
                          values[acts.RadialBoundsValue.MinR],
                          values[acts.RadialBoundsValue.MaxR], vol_id, lay_id))

        elif isinstance(bounds, acts.AnnulusBounds):
            rects.append((z_center - viz_thickness / 2.0, z_center + viz_thickness / 2.0,
                          values[acts.AnnulusBoundsValue.MinR],
                          values[acts.AnnulusBoundsValue.MaxR], vol_id, lay_id))

        elif isinstance(bounds, acts.RectangleBounds):
            half_x, half_y = values[0], values[1]
            local_corners = [
                (-half_x, -half_y), (half_x, -half_y),
                (half_x, half_y), (-half_x, half_y),
            ]
            rects.append(_planar_rect(surface, gctx, local_corners, vol_id, lay_id))

        elif isinstance(bounds, acts.TrapezoidBounds):
            half_x_neg_y, half_x_pos_y, half_y = values[0], values[1], values[2]
            local_corners = [
                (-half_x_neg_y, -half_y), (half_x_neg_y, -half_y),
                (half_x_pos_y, half_y), (-half_x_pos_y, half_y),
            ]
            rects.append(_planar_rect(surface, gctx, local_corners, vol_id, lay_id))

    return rects


def _planar_rect(surface, gctx, local_corners, vol_id, lay_id):
    center = surface.center(gctx)
    direction = surface.normal(gctx, center, acts.Vector3(0, 0, 1))
    z_vals, r_vals = [], []
    for lx, ly in local_corners:
        gp = surface.localToGlobal(gctx, acts.Vector2(lx, ly), direction)
        z_vals.append(gp[2])
        r_vals.append(np.sqrt(gp[0] ** 2 + gp[1] ** 2))
    return (min(z_vals), max(z_vals), min(r_vals), max(r_vals), vol_id, lay_id)


# ============================================================
# Data loading
# ============================================================

def load_hits(path):
    tree = uproot.open(path)["hits"]
    df = tree.arrays(["tx", "ty", "tz"], library="pd")
    x, y, z = df["tx"].to_numpy(), df["ty"].to_numpy(), df["tz"].to_numpy()
    return z, np.sqrt(x**2 + y**2)


def load_propagation(path):
    tree = uproot.open(path)["propagation_steps"]
    steps = tree.arrays(["g_z", "g_r", "approach_id", "boundary_id", "sensitive_id"], library="ak")
    return {
        "z": ak.flatten(steps["g_z"]).to_numpy(),
        "r": ak.flatten(steps["g_r"]).to_numpy(),
        "approach_id": ak.flatten(steps["approach_id"]).to_numpy(),
        "boundary_id": ak.flatten(steps["boundary_id"]).to_numpy(),
        "sensitive_id": ak.flatten(steps["sensitive_id"]).to_numpy(),
    }


# ============================================================
# Generic drawing helpers
# ============================================================

def draw_rectangles(ax, rects, label_fmt=None, linewidth=0.5, alpha=0.5, facecolor="skyblue"):
    """rects: iterable of (z_min, z_max, r_min, r_max, *extra_fields)."""
    for entry in rects:
        z_min, z_max, r_min, r_max = entry[:4]
        width, height = z_max - z_min, r_max - r_min
        ax.add_patch(patches.Rectangle(
            (z_min, r_min), width, height,
            linewidth=linewidth, edgecolor="black", facecolor=facecolor, alpha=alpha,
        ))
        if label_fmt is not None:
            ax.text(z_min + width / 2 - 2, r_min + height / 2, label_fmt(entry),
                     fontsize=4, ha="center", va="center", color="darkblue",
                     weight="bold", clip_on=True)

def draw_overlay(ax, z, r, mode="uniform", ids=None, n_bins=20, id_max=None, point_size=0.5, point_alpha=0.5):
    """Draws step/hit points, optionally colored by a discrete or bucketed ID array."""
    if mode == "uniform":
        ax.scatter(z, r, s=point_size, alpha=point_alpha, color="green", zorder=5)
        return None

    if mode == "discrete":
        # few distinct values -> one legend entry per value
        unique_ids = np.unique(ids)
        print(f"Distinct id values: {unique_ids}")
        cmap = plt.get_cmap("tab10")
        for i, uid in enumerate(unique_ids):
            mask = ids == uid
            ax.scatter(z[mask], r[mask], s=point_size, alpha=point_alpha,
                       color=cmap(i), zorder=5, label=f"id = {uid}")
        ax.legend(markerscale=10, fontsize=7, loc="upper right")
        return None

    if mode == "binned":
        # wide-range id -> bucket into n_bins colors on a continuous colormap
        id_max = id_max if id_max is not None else ids.max()
        bin_width = id_max / n_bins
        bucket = np.clip((ids // bin_width).astype(int), 0, n_bins - 1)
        sc = ax.scatter(z, r, s=point_size, alpha=point_alpha, c=bucket,
                         cmap="rainbow", vmin=0, vmax=n_bins - 1, zorder=5)
        return f"id bucket (width={bin_width:.0f})"

    raise ValueError(f"Unknown overlay mode: {mode}")


# ============================================================
# Main entry point
# ============================================================


def build_name(geometry, overlay, id_mode, generation):
    """Builds a consistent name like 'ODD_layer_propagation_sensitive_gen3',
    omitting overlay when plain and id_mode when uniform/not applicable."""
    parts = ["ODD", geometry]

    if overlay is not None:
        parts.append(overlay)
        # id_mode only makes sense for propagation overlays, and only
        # if it's not the (default, uninformative) "uniform" mode
        if overlay == "propagation" and id_mode != "uniform":
            parts.append(id_mode)

    parts.append(generation)
    return "_".join(parts)


def _planar_rect(surface, gctx, local_corners, vol_id, lay_id):
    center = surface.center(gctx)
    direction = surface.normal(gctx, center, acts.Vector3(0, 0, 1))
    z_vals, r_vals = [], []
    for lx, ly in local_corners:
        gp = surface.localToGlobal(gctx, acts.Vector2(lx, ly), direction)
        z_vals.append(gp[2])
        r_vals.append(np.sqrt(gp[0] ** 2 + gp[1] ** 2))
    return (min(z_vals), max(z_vals), min(r_vals), max(r_vals), vol_id, lay_id)



def make_plot(geometry, overlay=None, id_mode="uniform", generation=generation, filename=None, title=None):
    """
    geometry:   "volume" | "layer" | "sensitive" | "material"
    overlay:    None | "hits" | "propagation"
    id_mode: "uniform" | "approach" | "boundary" | "sensitive"
                (only used when overlay == "propagation")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- draw geometry rectangles ---
    if geometry == "volume":
        draw_rectangles(ax, volumeVisitor.volumes, label_fmt=lambda e: f"{e[5]}")
    elif geometry == "layer":
        draw_rectangles(ax, layerVisitor.layers, label_fmt=lambda e: f"{e[4]}.{e[5]}")
    elif (geometry == "sensitive"):
        draw_rectangles(ax, sensitiveVisitor.sensitives, linewidth=0.3)
    elif geometry == "material":
        draw_rectangles(ax, materialSurfaces, linewidth=0.3,
                        label_fmt=lambda e: f"{e[4]}.{e[5]}", facecolor="lightcoral")
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # --- draw overlay points ---
    cbar_label = None
    if overlay == "hits":
        z, r = load_hits(HITS_PATH)
        draw_overlay(ax, z, r, mode="uniform", point_size=1, point_alpha=0.3)
        ax.collections[-1].set_color("red")

    elif overlay == "propagation":
        steps = load_propagation(PROPAGATION_PATH)
        z, r = steps["z"], steps["r"]

        if id_mode == "uniform":
            draw_overlay(ax, z, r, mode="uniform", point_size=0.1, point_alpha=0.1)
        elif id_mode == "approach":
            draw_overlay(ax, z, r, mode="discrete", ids=steps["approach_id"])
        elif id_mode == "boundary":
            draw_overlay(ax, z, r, mode="discrete", ids=steps["boundary_id"])
        elif id_mode == "sensitive":
            cbar_label = draw_overlay(ax, z, r, mode="binned", ids=steps["sensitive_id"], n_bins=20, id_max=2500)

    if cbar_label is not None:
        plt.colorbar(ax.collections[-1], ax=ax, label=cbar_label)

    name = build_name(geometry, overlay, id_mode, generation)

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(title or name.replace("_", " "))
    ax.autoscale_view()

    # Route output into a per-generation directory, e.g.
    # /home/lea-baumann/Documents/PhD/ACTS/Plot_Results_Gen1 for generation == "gen1".
    plot_dir = pathlib.Path("/home/lea-baumann/Documents/PhD/ACTS") / f"Plot_Results_{generation.capitalize()}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    filepath = pathlib.Path(filename) if filename else pathlib.Path(f"{name}.png")
    if not filepath.is_absolute():
        filepath = plot_dir / filepath

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")


# ============================================================
# Setup (run once)
# ============================================================

HITS_PATH = "/home/lea-baumann/Documents/PhD/ACTS/Outputs/odd_output_propagation_events1000/hits.root"
PROPAGATION_PATH = "/home/lea-baumann/Documents/PhD/ACTS/acts/propagation_"+generation+"/propagation_steps.root"

gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
volumeVisitor = LeasVisitor(gctx)
layerVisitor = LeasLayerVisitor(gctx)
sensitiveVisitor = LeasSensitiveVisitor(gctx)

trackingGeometry.apply(volumeVisitor)
trackingGeometry.apply(layerVisitor)
trackingGeometry.apply(sensitiveVisitor)

# Surfaces carrying an ISurfaceMaterial (approach/representing surfaces, beam pipe,
# and any sensor with mapped material), taken from the ready-made helper.
materialSurfaces = build_material_rects(trackingGeometry.extractMaterialSurfaces(), gctx)

print(f"Volumes: {len(volumeVisitor.volumes)} | Layers: {layerVisitor.num_layers} | "
      f"Sensitive: {len(sensitiveVisitor.sensitives)} (skipped: {sensitiveVisitor.n_skipped}) | "
      f"Material surfaces: {len(materialSurfaces)}")


# ============================================================
# Generate the plots
# ============================================================

#make_plot("volume")
#make_plot("volume", overlay="hits")
#make_plot("volume", overlay="propagation")

#make_plot("layer")
#make_plot("layer", overlay="hits")
#make_plot("layer", overlay="propagation")

#make_plot("sensitive")
#make_plot("sensitive", overlay="hits")
#make_plot("sensitive", overlay="propagation", id_mode="sensitive")
#make_plot("sensitive", overlay="propagation", id_mode="approach")
#make_plot("sensitive", overlay="propagation", id_mode="boundary")
#make_plot("sensitive", overlay="propagation", id_mode="uniform")

make_plot("material")
#make_plot("material", overlay="hits")
#make_plot("material", overlay="propagation")


if args.reco:
    addSeeding(
        s,
        trackingGeometry,
        field,
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 * u.e / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 * u.e / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1.0] * 6,
        particleHypothesis=acts.ParticleHypothesis.muon,
        geoSelectionConfigFile=oddSeedingSel,
        outputDirRoot=outputDir if args.output_root else None,
        outputDirCsv=outputDir if args.output_csv else None,
    )

    if seedFilter_ML:
        addSeedFilterML(
            s,
            SeedFilterMLDBScanConfig(
                epsilonDBScan=0.03, minPointsDBScan=2, minSeedScore=0.1
            ),
            onnxModelFile=os.path.dirname(__file__)
            + "/MLAmbiguityResolution/seedDuplicateClassifier.onnx",
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
        )

    addCKFTracks(
        s,
        trackingGeometry,
        field,
        TrackSelectorConfig(
            pt=(1.0 * u.GeV if args.ttbar else 0.0, None),
            absEta=(None, 3.0),
            loc0=(-4.0 * u.mm, 4.0 * u.mm),
            nMeasurementsMin=7,
            maxHoles=2,
            maxOutliers=2,
        ),
        CkfConfig(
            chi2CutOffMeasurement=15.0,
            chi2CutOffOutlier=25.0,
            numMeasurementsCutOff=2,
            seedDeduplication=True,
            stayOnSeed=True,
            pixelVolumes=[16, 17, 18],
            stripVolumes=[23, 24, 25],
            maxPixelHoles=1,
            maxStripHoles=2,
            constrainToVolumes=[
                2,  # beam pipe
                32,
                4,  # beam pip gap
                16,
                17,
                18,  # pixel
                20,  # PST
                23,
                24,
                25,  # short strip
                26,
                8,  # long strip gap
                28,
                29,
                30,  # long strip
            ],
        ),
        outputDirRoot=outputDir if args.output_root else None,
        outputDirCsv=outputDir if args.output_csv else None,
        writeCovMat=True,
    )

    if ambi_ML:
        addAmbiguityResolutionML(
            s,
            AmbiguityResolutionMLConfig(
                maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7
            ),
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
            onnxModelFile=os.path.dirname(__file__)
            + "/MLAmbiguityResolution/duplicateClassifier.onnx",
        )

    elif ambi_scoring:
        addScoreBasedAmbiguityResolution(
            s,
            ScoreBasedAmbiguityResolutionConfig(
                minScore=0,
                minScoreSharedTracks=1,
                maxShared=2,
                minUnshared=3,
                maxSharedTracksPerMeasurement=2,
                useAmbiguityScoring=False,
            ),
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
            ambiVolumeFile=ambi_config,
            writeCovMat=True,
        )
    else:
        addAmbiguityResolution(
            s,
            AmbiguityResolutionConfig(
                maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7
            ),
            outputDirRoot=outputDir if args.output_root else None,
            outputDirCsv=outputDir if args.output_csv else None,
            writeCovMat=True,
        )

    addVertexFitting(
        s,
        field,
        vertexFinder=VertexFinder.AMVF,
        outputDirRoot=outputDir if args.output_root else None,
        outputDirCsv=outputDir if args.output_csv else None,
    )

if args.output_parquet:
    try:
        from acts.arrow import (
            particleSchema,
            simHitSchema,
            trackSchema,
        )
        from acts.examples.arrow import (
            ArrowParticleOutputConverter,
            ArrowSimHitOutputConverter,
            ArrowTrackOutputConverter,
            makeVolumeIdDetectorResolver,
            ParquetWriter,
        )
    except ImportError as e:
        raise RuntimeError(
            "Parquet output requested but acts.examples.arrow is not available; "
            "rebuild with ACTS_BUILD_EXAMPLES_PARQUET=ON."
        ) from e

    # ODD volume -> detector enum mapping used in parquet simhit export.
    # 0..8 are subdetector-specific enums; 255 marks unknown/unmatched.
    _odd_detector_resolver = makeVolumeIdDetectorResolver(
        {
            7: 0,  # pixel_neg_endcap
            8: 1,  # pixel_barrel
            9: 2,  # pixel_pos_endcap
            12: 3,  # short_neg_endcap
            13: 4,  # short_barrel
            14: 5,  # short_pos_endcap
            16: 6,  # long_neg_endcap
            17: 7,  # long_barrel
            18: 8,  # long_pos_endcap
        },
        255,
    )

    # Each converter parks an arrow::Table on the whiteboard under a fresh
    # key, and one ParquetWriter picks them all up.
    arrParticleConv = ArrowParticleOutputConverter(
        level=acts.logging.INFO,
        inputParticles="particles_simulated",
        outputTable="particles_arrow",
    )
    s.addAlgorithm(arrParticleConv)

    arrSimHitConv = ArrowSimHitOutputConverter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        inputParticles="particles_simulated",
        inputClusters="clusters",
        inputSimHitMeasurementsMap="simhit_measurements_map",
        outputTable="simhits_arrow",
        detectorResolver=_odd_detector_resolver,
    )
    s.addAlgorithm(arrSimHitConv)

    if args.reco:
        arrTrackConv = ArrowTrackOutputConverter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputTrackParticleMatching="track_particle_matching",
            inputParticles="particles_simulated",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            outputTable="tracks_arrow",
        )
        s.addAlgorithm(arrTrackConv)

    s.addWriter(
        ParquetWriter(
            level=acts.logging.INFO,
            outputDir=str(outputDir),
            collections={
                arrSimHitConv.config.outputTable: "simhits",
                arrTrackConv.config.outputTable: "tracks",
                arrParticleConv.config.outputTable: "particles",
            },
            expectedSchemas={
                arrSimHitConv.config.outputTable: simHitSchema(),
                arrTrackConv.config.outputTable: trackSchema(),
                arrParticleConv.config.outputTable: particleSchema(),
            },
        )
    )

s.run()
