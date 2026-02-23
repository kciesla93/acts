// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// @file HoughTransformSeeder.hpp
// @author Riley Xu then modified to ACTS by Jahred Adelman
// @brief Implements track-seeding using a Hough transform.
//
// Using the Lorentz force equation, one can relate the phi of a track and the
// coordinate of a single hit:
//
//      A * q / pT = sin(phi_track - phi_hit) / r
//
// where
//      A   : 3 * 10^-4 GeV / (c*mm*e) for the ATLAS B field (to be configured)
//      q   : charge of the particle
//      pT  : transverse momentum
//      r   : cylindrical radius of the hit from the beamline
//      phi : in radians
//
// Here, q/pT and phi_track (ie at phi at perigee) are unknown. This equation
// forms a line in q/pT vs phi_track spac since the sin function above can be
// approximated with sin(x) ~ x. Each hit will have its own line based on its
// position (phi and r). However, note that hits belonging to the same track
// will have lines that intersect at the track's q/pT and phi. In this manner,
// we can conduct pattern -matching by looking for intersections of these pT-phi
// lines.
//
// In other words, given some assumed q/pT for the track one can take the phi
// and r for a hit and convert that to what the phi(perigee) for a track must
// have been. We loop over the q/pT bins, and at the true value all the hits
// should line up in the same bin
//
// To easily find intersections, we first pixelate (equivalently, we make a 2d
// histogram from) the graph of all the hit's lines in q/pT vs phi_track space.
// We then apply a convolution (i.e. a scanning window) to pick out points with
// multiple lines going through them. These points become our seed.
//
// In principle the Hough transform can be used for an entire region (i.e. .2
// phi x .2 eta) or larger. However this can lead to an excessive number of
// hits/lines in the transform houghHist, leading to spurious intersections.
// Instead, we can use multiple transforms that each cover a slice in z0, and
// simply combine all the seeds found. These are the subregions
//
// References:
//      Martensson Thesis:
//      http://uu.diva-portal.org/smash/get/diva2:1341509/FULLTEXT01.pdf
//

// We adopt the following nomenclature within this class:
//      houghHist: The 'graph' in q/pT vs phi_track space, filled with a line
//      calculated as above for each hit. point: A specific q/pT and phi_track
//      bin in the above houghHist; i.e. what is normally called a pixel
//             but I don't want to confuse this with the detector type. A
//             point's value is the number of lines that go through it.
//
// For the first iteration, x refers to phi_track, and y refers to q/pT,
// although this should remain flexible. These are set via the variables m_par_x
// and m_par_y.
//
// NOTE: y=0 represents the lowest q/pT bin, x=0 represents the lowest
// phi(perigee) bin
//      houghHist[y=0][x=0]      : lowest q/pT and lowest phi_track bin
//      houghHist[y=size-1][x=0] : highest q/pT and lowest phi_track bin
//

#pragma once

#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Seeding/HoughTransformUtils.hpp"
#include "Acts/Utilities/Delegate.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <cstddef>
#include <format>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <TFile.h>
#include <TH2.h>
#include <TTree.h>

namespace ActsExamples {
struct AlgorithmContext;
}  // namespace ActsExamples

namespace ActsExamples {
struct HoughMeasurementStruct;
}  // namespace ActsExamples

using ResultDouble = Acts::Result<double>;
using ResultBool = Acts::Result<bool>;
using ResultUnsigned = Acts::Result<unsigned>;

using FieldCorrector = Acts::Delegate<ResultDouble(
    unsigned, double, double)>;  // (unsigned region, double y, double r)
using LayerIDFinder = Acts::Delegate<ResultUnsigned(
    double)>;  // (double r) this function will map the r of a measurement to a
               // layer.
using SliceTester = Acts::Delegate<ResultBool(
    const std::shared_ptr<ActsExamples::HoughMeasurementStruct>&,
    int)>;  // (double eta, int slice) returns
            // true if measurement in slice

namespace Acts {
class TrackingGeometry;
}

namespace ActsExamples {
/// Used in multiple places. The 2d vector refers to the 2d houghHist. For a
/// single layer, the int refers to the number of hits in the bin of the
/// houghHist
//// For the total houghHist, the int counts the number of layers with one or
/// more
/// hit in that bin
// The unsigned is a counter that will point to a spacepoint or to a measurement
// object

/// An houghHist is a 2d array of points, where each point has a value.
/// The value starts as the number of hit layers, but can change with effects
/// like a convolution. Also stored are indices of all hits that contributed to
/// each bin. Size m_houghHistSize_y * m_houghHistSize_x. (NOTE y is row
/// coordinate) For now, what is stored is actually the index of the object in
/// the vectors, so we can get the Index layer
using HoughMeasurement = unsigned;  // measurement index
using HoughHist = Acts::HoughTransformUtils::HoughPlane<HoughMeasurement>;

enum HoughHitType { SP = 0, MEASUREMENT = 1 };
enum class Binning { EqudistantQoverPt, EqudistantPt, Steps, FinerCentral };
enum class Slicing { None, EqudistantEta, Wedges };

/// The measurements and SP are ugly to use, this is a convenience struct that
/// contains the needed information
struct HoughMeasurementStruct {
  unsigned layer;
  double phi;
  double radius;
  double z;
  double eta;
  std::vector<Index> indices;
  HoughHitType type;
  HoughMeasurementStruct(unsigned l, double p, double r, double thez,
                         double theeta, std::vector<Index>& i, HoughHitType t)
      : layer(l),
        phi(p),
        radius(r),
        z(thez),
        eta(theeta),
        indices(i),
        type(t) {}
};

/// Construct track seeds from space points.
class HoughTransformSeeder final : public IAlgorithm {
 public:
  struct Config {
    /// Input space point collections.
    ///
    /// We allow multiple space point collections to allow different parts of
    /// the detector to use different algorithms for space point construction,
    /// e.g. single-hit space points for pixel-like detectors or double-hit
    /// space points for strip-like detectors.
    /// Note that we don't *need* spacepoints (measurements can be used instead)
    std::vector<std::string> inputSpacePoints;
    /// Output track seed collection.
    std::string outputSeeds;
    /// Output hough track collection.
    std::string outputProtoTracks;
    /// Tracking geometry required to access global-to-local transforms.
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry;
    /// For which part of the detector geometry should space points be created.
    ///
    /// Only volumes and layers can be set. Zero values can be used as wildcards
    /// to select larger parts of the hierarchy, i.e. setting only the volume
    /// selects all measurements within that volume. Adding a single identifier
    /// with all components set to zero selects all available measurements. The
    /// selection must not have duplicates.
    std::vector<Acts::GeometryIdentifier> geometrySelection;

    /// Input measurements collection.
    std::string inputMeasurements;

    // Subregions are ways to divide up hits for the Hough Transform. Just as
    // one simple example, one may consider that hits with z < 50 mm belong to
    // one subregion, and hits with z > -50 mm belong to a second subregion.
    // Note that hits even in this toy example belong to more than one
    // subregion. But since not all hits are considered this provides a way to
    // reduce potential combinatorics

    std::vector<int> subRegions = {-1, 0, 1, 2, 3,  4,  5,
                                   6,  7, 8, 9, 10, 11, 12};

    unsigned nLayers = 10;  // total number of layers

    float xMin = -std::numbers::pi;  // minphi
    float xMax = std::numbers::pi;   // maxphi
    float yMin = -1.;                // min q/pt, -1/1 GeV
    float yMax = 1.;                 // max q/pt, +1/1 GeV

    /// Size of the houghHists. One obvious concern with this being too big is
    /// that it will take up more memory But the bins of the houghHist are
    /// looped over in two places as well, so there are CPU performance issues,
    /// too The first is just when finding the bins and hits that provide
    /// candidates, so the loop is over x*y bins. The other is when filling the
    /// bins. The loop is over y bins, and for each y bin we find the min and
    /// max x for each hit

    unsigned houghHistSize_x = 7000;  // i.e. number of bins in phi_track
    unsigned houghHistSize_y = 216;   // i.e. number of bins in q/pT

    /// For each assumed q/pT (y) we find the appropriate phi (x) bin for a hit.
    /// But if extend = 2 (for example) We then fill in addition 2 bins to the
    /// right, and 2 bins to the left, so 5 bins and not just 1
    std::vector<unsigned> hitExtend_x = {
        1, 1, 0, 0, 0, 0,
        0, 0, 0, 0};  // Hit lines will fill extra bins in x by this amount on
                      // each side, size == nLayers

    /// === Seeds for Hough ==
    std::vector<int> threshold = {
        9};  // Minimum number of measurements per bin to accept as a
             // prototrack/seed. Right now this is a single number, can be
             // expanded in the future if we want to be more clever

    int localMaxWindowSize = 0;  // Only create candidates from a local maximum

    double kA = 0.0003;  // Assume B = 2T constant. Can apply corrections to
                         // this with fieldCorrection function
                         // This 3e-4 comes from the 2T field when converted to
                         // units of GeV / (c*mm*e)

    // it's up to the user to connect these to the functions they want to use
    FieldCorrector fieldCorrector;
    LayerIDFinder layerIDFinder;
    SliceTester sliceTester;

    Acts::HoughTransformUtils::HoughPlaneConfig plane{houghHistSize_y,
                                                      houghHistSize_x};
    Acts::HoughTransformUtils::PeakFinders::SlidingWindowConfig slidingWindow{
        .threshold = 6,
        .xWindowSize = 16,
        .yWindowSize = 16,
        .recenter = true,
        .xRecenterSize = 3,
        .yRecenterSize = 3};

    std::uint32_t truthThreshold = 5;

    Binning binning = Binning::FinerCentral;
    Slicing slicing = Slicing::Wedges;

    bool writeToSingleFile = false;  // Defaults to false for now
  };

  /// Construct the seeding algorithm.
  ///
  /// @param cfg is the algorithm configuration
  /// @param lvl is the logging level
  HoughTransformSeeder(Config cfg, Acts::Logging::Level lvl);

  /// Run the seeding algorithm.
  ///
  /// @param txt is the algorithm context with event information
  /// @return a process code indication success or failure
  ProcessCode execute(const AlgorithmContext& ctx) const final;
  ProcessCode finalize() final;

  /// Const access to the config
  const Config& config() const { return m_cfg; }

  double getMinX() const { return m_cfg.xMin; }
  double getMaxX() const { return m_cfg.xMax; }
  double getMinY() const { return m_cfg.yMin; }
  double getMaxY() const { return m_cfg.yMax; }
  unsigned getThreshold() const {
    return m_cfg.threshold[0];  // for now this is just one number in the
                                // vector, can be more in the future
  }
  const std::vector<int>& getSubRegions() const { return m_cfg.subRegions; }

  double yToX(double y, double r,
              double phi) const;  // calculate the hough equation

 private:
  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;
  const Acts::Logger& logger() const { return *m_logger; }

  WriteDataHandle<ProtoTrackContainer> m_outputProtoTracks{this,
                                                           "OutputProtoTracks"};
  std::vector<std::unique_ptr<ReadDataHandle<SimSpacePointContainer>>>
      m_inputSpacePoints{};

  ReadDataHandle<MeasurementContainer> m_inputMeasurements{this,
                                                           "InputMeasurements"};

  ReadDataHandle<MeasurementParticlesMap> m_inputMeasurementParticlesMap{
      this, "measurement_particles_map"};

  ReadDataHandle<SimParticleContainer> m_inputParticles{this,
                                                        "particles_simulated"};

  ////////////////////////////////////////////////////////////////////////
  /// Convenience

  double m_step_x;  // step size of the bin boundaries in x
  double m_step_y;  // step size of the bin boundaries in y
  /// Bin boundaries, where m_bins_x[i] is the lower bound of bin i.
  /// These are calculated from m_xMin/m_xMax
  std::vector<double> m_bins_x;  // size == m_houghHistSize_x + 1.
  std::vector<double> m_bins_y;  // size == m_houghHistSize_y + 1

  ///////////////////////////////////////////////////////////////////////
  // Core functions, the second/ one calls the first one per layer
  HoughHist createHoughHist(int subregion) const;

  ///////////////////////////////////////////////////////////////////////
  // Helpers
  std::pair<unsigned, unsigned> yToXBins(std::size_t yBin_min,
                                         std::size_t yBin_max, double r,
                                         double phi,
                                         unsigned layer)
      const;  // given y bins, return x bins passed that need to be filled in
              // the HoughHist, including extensions

  unsigned getExtension(unsigned y, unsigned layer) const;  // return extensions
  bool passThreshold(HoughHist const& houghHist, unsigned x,
                     unsigned y) const;  // did we pass extensions?
  void drawHoughHist(HoughHist const& houghHist,
                     std::string const& name);  // for making pretty plots
  std::vector<std::vector<int>> getComboIndices(std::vector<std::size_t>& sizes)
      const;  // useful to find all candidates from given bins that pass
              // (looping over hit combinatorics)

  // functions to clean up the code and convert SPs and measurements to the
  // HoughMeasurement format
  void addMeasurements(const AlgorithmContext& ctx) const;
  void addSpacePoints(const AlgorithmContext& ctx) const;

  struct Writer;
  std::unique_ptr<Writer> m_writer;
};

struct ActsExamples::HoughTransformSeeder::Writer {
  TFile* file_truth = nullptr;
  TFile* file_histo = nullptr;
  TTree* tree = nullptr;
  std::mutex writer_mutex;

  std::uint64_t event_number{};
  int slice{};
  std::uint32_t bin_qOverPt{};
  std::uint32_t bin_phi{};
  std::uint64_t truth_hash{};
  std::uint32_t truth_hits{};

  explicit Writer(bool use_single_file)
      : file_truth(TFile::Open("truth.root", "recreate")) {
    if (use_single_file) {
      file_histo = TFile::Open("out.root", "recreate");
    }

    tree = new TTree("truth", "truth");
    tree->SetDirectory(file_truth);

    tree->Branch("event_number", &event_number);
    tree->Branch("slice", &slice);
    tree->Branch("bin_qOverPt", &bin_qOverPt);
    tree->Branch("bin_phi", &bin_phi);
    tree->Branch("truth_hash", &truth_hash);
    tree->Branch("truth_hits", &truth_hits);
  }

  Writer(const Writer&) = delete;
  Writer operator=(const Writer&) = delete;

  void writeTree(std::uint64_t eventNumber, int sliceId,
                 std::uint32_t qOverPt_bin, std::uint32_t phi_bin,
                 std::uint64_t particle_hash, std::uint32_t nHits) {
    std::scoped_lock guard(writer_mutex);

    event_number = eventNumber;
    slice = sliceId;
    bin_qOverPt = qOverPt_bin;
    bin_phi = phi_bin;
    truth_hash = particle_hash;
    truth_hits = nHits;

    tree->Fill();
  }

  template <class T>
  void writeObj(T* obj) {
    std::scoped_lock guard(writer_mutex);

    file_histo->WriteObject(obj, obj->GetName());
  }

  template <class T>
  static void writeObjThread(T* obj) {
    const auto thread_id_hash =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto file = TFile::Open(std::format("out_{}.root", thread_id_hash).c_str(),
                            "update");
    file->WriteObject(obj, obj->GetName());
    file->Close();
  }

  void close() {
    file_truth->Write();
    file_truth->Close();

    if (file_histo != nullptr) {
      file_histo->Write();
      file_histo->Close();
    }
  }
};

namespace Wedges {
struct Reg {
  float center;
  float width;
};

struct Wedge {
  Reg phi;
  float aleft;
  float aright;
  float bleft;
  float bright;

  Wedge(Reg p, Reg z, Reg eta) : phi(p) {
    aleft = std::tan(2.0 * std::atan(std::exp(-(eta.center - eta.width))));
    aright = std::tan(2.0 * std::atan(std::exp(-(eta.center + eta.width))));
    bleft = -aleft / (z.center - z.width);
    bright = -aright / (z.center + z.width);
  }

  static float delta_phi(float phi1, float phi2) {
    const float delta = phi1 - phi2;
    if (delta > M_PI) {
      return delta - M_PI;
    } else if (delta < 0) {
      return delta + M_PI;
    }

    return delta;
  }

  bool in_rPhiZ(float r, float p, float z) const {
    if (std::fabs(delta_phi(p, phi.center)) > phi.width) {
      return false;
    }

    if (aleft > 0 && aright > 0) {
      return aleft * z + bleft > r && r > aright * z + bright;
    } else if (aleft < 0 && aright > 0) {
      return aleft * z + bleft < r && r > aright * z + bright;
    }
    return aleft * z + bleft < r && r < aright * z + bright;
  }
};

static constexpr std::size_t nWedges = 13;
static constexpr float etaWidth = 0.23076923076923078;
static constexpr float zWidth = 150;  // [mm]
static constexpr Reg phi{0, M_PI};
static constexpr Reg z{0, 1. / zWidth};

static std::array<Wedge, nWedges> wedges{{
    {phi, z, {-2.769230769230769, etaWidth}},
    {phi, z, {-2.3076923076923075, etaWidth}},
    {phi, z, {-1.8461538461538458, etaWidth}},
    {phi, z, {-1.3846153846153846, etaWidth}},
    {phi, z, {-0.9230769230769229, etaWidth}},
    {phi, z, {-0.4615384615384613, etaWidth}},
    {phi, z, {0, etaWidth}},
    {phi, z, {0.4615384615384616, etaWidth}},
    {phi, z, {0.9230769230769234, etaWidth}},
    {phi, z, {1.384615384615385, etaWidth}},
    {phi, z, {1.8461538461538467, etaWidth}},
    {phi, z, {2.3076923076923084, etaWidth}},
    {phi, z, {2.769230769230769, etaWidth}},
}};
}  // namespace Wedges

}  // namespace ActsExamples
