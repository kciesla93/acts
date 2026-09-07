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
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Seeding/HoughTransformUtils.hpp"
#include "Acts/Utilities/Delegate.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "Acts/Utilities/ScopedTimer.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Seed.hpp"
#include "ActsExamples/EventData/SpacePoint.hpp"
#include "ActsExamples/EventData/TruthMatching.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/Io/Csv/CsvInputOutput.hpp"

#include <cstddef>
#include <format>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <ROOT/RCsvDS.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TH2.h>
#include <TTree.h>

namespace ActsExamples {

/// Used in multiple places. The 2d vector refers to the 2d houghHist. For a
/// single layer, the int refers to the number of hits in the bin of the
/// houghHist
//// For the total houghHist, the int counts the number of layers with one or
/// more
/// hit in that bin
// The unsigned is a counter that will point to a space point or to a
// measurement object

/// An houghHist is a 2d array of points, where each point has a value.
/// The value starts as the number of hit layers, but can change with effects
/// like a convolution. Also stored are indices of all hits that contributed to
/// each bin. Size m_houghHistSize_y * m_houghHistSize_x. (NOTE y is row
/// coordinate) For now, what is stored is actually the index of the object in
/// the vectors, so we can get the Index layer
using HoughMeasurement = std::uint64_t;  // measurement index
using HoughHist = Acts::HoughTransformUtils::HoughPlane<HoughMeasurement>;

enum HoughHitType { SP = 0, MEASUREMENT = 1 };
enum class Binning { EqudistantQoverPt, EqudistantPt, Steps, FinerCentral };
enum class Slicing { None, Eta };
enum class SeedType {
  All,
  NearTriplet,
  FarTriplet,
  NearTripletSkipFirst,
  NearMiddleFarTriplet,
};

/// The measurements and SP are ugly to use, this is a convenience struct that
/// contains the needed information
struct HoughMeasurementStruct {
  unsigned layer;
  double phi;
  double radius;
  double z;
  double eta;
  std::vector<Index> indices;
  std::uint32_t sp_index{};
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
  using ResultDouble = Acts::Result<double>;
  using ResultBool = Acts::Result<bool>;
  using ResultUnsigned = Acts::Result<unsigned>;
  using ResultFiltering =
      Acts::Result<std::vector<const HoughMeasurementStruct*>>;

  using FieldCorrector = Acts::Delegate<ResultDouble(
      unsigned, double, double)>;  // (unsigned region, double y, double r)
  using LayerIDFinder = Acts::Delegate<ResultUnsigned(
      double)>;  // (double r) this function will map the r of a measurement to
                 // a layer.
  using SliceTester =
      Acts::Delegate<ResultBool(const std::shared_ptr<HoughMeasurementStruct>&,
                                int)>;  // (double z,unsigned layer, int slice)
                                        // returns true if measurement in slice
  using SeedFilter = Acts::Delegate<ResultFiltering(
      std::span<const HoughMeasurementStruct*>)>;

  struct Config {
    /// Input space point collections.
    std::string inputSpacePoints;
    /// Output track seed collection.
    std::string outputSeeds;
    /// Output hough track collection.
    std::string outputProtoTracks;
    /// Tracking geometry required to access global-to-local transforms.
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry;
    /// Magnetic field
    std::shared_ptr<const Acts::MagneticFieldProvider> bField;
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
    // reduce potential combinatorics.

    /// Number of subregions to split HT into if slicing other than
    /// Slicing::None is provided.
    std::size_t nSubRegions = 0;

    /// Subregions to use. If nothing is provided it will be set to {-1} (entire
    /// detector) for Slicing::None or {0..nSubRegions-1} for other slicing
    /// method. Useful for selecting subset of subregions.
    std::vector<int> subRegions = {};

    unsigned nLayers = 48;  // total number of layers

    float xMin = -std::numbers::pi;  // minphi
    float xMax = std::numbers::pi;   // maxphi
    float yMin = -1. / 0.9;          // min q/pt, -900 MeV
    float yMax = 1. / 0.9;           // max q/pt, +900 MeV

    /// Size of the houghHists. One obvious concern with this being too big is
    /// that it will take up more memory But the bins of the houghHist are
    /// looped over in two places as well, so there are CPU performance issues,
    /// too The first is just when finding the bins and hits that provide
    /// candidates, so the loop is over x*y bins. The other is when filling the
    /// bins. The loop is over y bins, and for each y bin we find the min and
    /// max x for each hit

    unsigned houghHistSize_x = 7000;  // i.e. number of bins in phi_track
    unsigned houghHistSize_y = 256;   // i.e. number of bins in q/pT

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
             // proto track/seed. Right now this is a single number, can be
             // expanded in the future if we want to be more clever

    int localMaxWindowSize = 0;  // Only create candidates from a local maximum

    // it's up to the user to connect these to the functions they want to use
    FieldCorrector fieldCorrector;
    LayerIDFinder layerIDFinder;
    SliceTester sliceTester;
    SeedFilter seedFilter;

    Acts::HoughTransformUtils::HoughPlaneConfig plane{houghHistSize_y,
                                                      houghHistSize_x};
    Acts::HoughTransformUtils::PeakFinders::SlidingWindowConfig slidingWindow{
        .threshold = 6,
        .xWindowSize = 16,
        .yWindowSize = 16,
        .recenter = true,
        .xRecenterSize = 3,
        .yRecenterSize = 3};

    std::uint32_t truthHoughThreshold = 3;
    std::uint32_t truthNhits = 7;
    float truthDzCut = 150;  // [mm]
    float truthPtCut = 1;    // [GeV]

    static constexpr Binning binning = Binning::FinerCentral;
    static constexpr Slicing slicing = Slicing::Eta;
    static constexpr SeedType seedType = SeedType::NearMiddleFarTriplet;

    bool writeToSingleFile = false;  // Defaults to false for now
  };

  /// Construct the seeding algorithm.
  ///
  /// @param cfg is the algorithm configuration
  /// @param lvl is the logging level
  explicit HoughTransformSeeder(
      const Config& cfg, std::unique_ptr<const Acts::Logger> logger = nullptr);

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

  ReadDataHandle<MeasurementContainer> m_inputMeasurements{this,
                                                           "InputMeasurements"};

  ReadDataHandle<MeasurementParticlesMap> m_inputMeasurementParticlesMap{
      this, "measurement_particles_map"};

  ReadDataHandle<SimParticleContainer> m_inputParticles{this, "TruthParticles"};

  ReadDataHandle<SpacePointContainer> m_inputSpacePoints{this,
                                                         "InputSpacePoints"};

  WriteDataHandle<ProtoTrackContainer> m_outputProtoTracks{this,
                                                           "OutputProtoTracks"};

  WriteDataHandle<SeedContainer> m_outputSeeds{this, "OutputSeeds"};

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
  void fillHoughHist(HoughHist& houghHist, int subregion,
                     Acts::AveragingScopedTimer& timer) const;

  ///////////////////////////////////////////////////////////////////////
  // Helpers
  std::pair<unsigned, unsigned> yToXBins(std::size_t yBin_min,
                                         std::size_t yBin_max, double r,
                                         double phi,
                                         unsigned layer)
      const;  // given y bins, return x bins passed that need to be filled in
              // the HoughHist, including extensions

  unsigned getExtension(unsigned y, unsigned layer) const;  // return extensions
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

  struct NNReader;
  std::unique_ptr<NNReader> m_reader;

  double m_bFieldZ;
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
  std::vector<std::uint64_t> truth_hashes{};
  std::vector<std::uint32_t> truth_counts{};

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
    tree->Branch("truth_hashes", &truth_hashes);
    tree->Branch("truth_counts", &truth_counts);
  }

  Writer(const Writer&) = delete;
  Writer operator=(const Writer&) = delete;

  void writeTree(std::uint64_t eventNumber, int sliceId,
                 std::uint32_t qOverPt_bin, std::uint32_t phi_bin,
                 std::uint64_t particle_hash, std::uint32_t nHits,
                 const std::map<std::uint64_t, std::uint32_t>& all_counts) {
    std::scoped_lock guard(writer_mutex);

    truth_hashes.clear();
    truth_counts.clear();

    event_number = eventNumber;
    slice = sliceId;
    bin_qOverPt = qOverPt_bin;
    bin_phi = phi_bin;
    truth_hash = particle_hash;
    truth_hits = nHits;

    for (const auto& [hash, count] : all_counts) {
      truth_hashes.push_back(hash);
      truth_counts.push_back(count);
    }

    tree->Fill();
  }

  template <class T>
  void writeObj(T* obj) {
    std::scoped_lock guard(writer_mutex);

    file_histo->WriteObject(obj, obj->GetName());
  }

  template <class T>
  static void writeObjThread(T* obj, std::size_t threadId) {
    auto file =
        TFile::Open(std::format("out_{}.root", threadId).c_str(), "update");
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

struct ActsExamples::HoughTransformSeeder::NNReader {
  using Peak = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
  using EventPeak = std::array<std::uint32_t, 5>;

  std::vector<EventPeak> allPeaks;

  explicit NNReader(std::string_view filename) {
    namespace fs = std::filesystem;
    const fs::path csv(filename);

    if (!fs::exists(csv)) {
      return;
    }

    auto reader = ActsExamples::CsvReader(filename.data());
    std::vector<std::string> columns;
    while (reader.read(columns)) {
      if (Acts::rangeContainsValue(columns, "event_id")) {
        continue;
      }

      std::ranges::transform(columns, allPeaks.emplace_back().data(),
                             [](const std::string& s) { return std::stoi(s); });
    }
  }

  NNReader(const NNReader&) = delete;
  NNReader operator=(const NNReader&) = delete;

  /// Check if peaks are loaded
  [[nodiscard]] bool hasPeaks() const noexcept { return !allPeaks.empty(); }

  std::vector<Peak> getPeaks(std::uint32_t eventNumber,
                             std::uint32_t slice) const noexcept {
    std::vector<Peak> peaks;
    std::ranges::for_each(allPeaks,
                          [&peaks, eventNumber, slice](EventPeak peak) {
                            if (peak[0] == eventNumber && peak[1] == slice) {
                              peaks.emplace_back(peak[2], peak[3], peak[4]);
                            }
                          });
    return peaks;
  }
};

namespace EtaSlicer {
struct Reg {
  float center;
  float width;
};

struct EtaSlice {
  float aleft;
  float aright;
  float bleft;
  float bright;

  EtaSlice(Reg z, Reg eta) {
    aleft = std::tan(2.0 * std::atan(std::exp(-(eta.center - eta.width))));
    aright = std::tan(2.0 * std::atan(std::exp(-(eta.center + eta.width))));
    bleft = aleft * z.width;
    bright = -aright * z.width;
  }

  bool in_rZ(float r, float z) const {
    const float lhs = aleft * z + bleft;
    const float rhs = aright * z + bright;

    if (aleft > 0 && aright > 0) {
      return lhs > r && r > rhs;
    } else if (aleft < 0 && aright > 0) {
      return lhs < r && r > rhs;
    }

    return lhs < r && r < rhs;
  }
};

std::vector<EtaSlice> make_slices(std::size_t nSlices) {
  static constexpr float zCenter = 0;   // [mm]
  static constexpr float zWidth = 150;  // [mm]
  static constexpr Reg z{zCenter, zWidth};
  const double etaWidth = 3. / nSlices;

  std::vector<EtaSlice> slices;
  slices.reserve(nSlices);
  for (std::size_t iSlice = 0; iSlice < nSlices; ++iSlice) {
    slices.emplace_back(z, Reg(-3. + (etaWidth * (2 * iSlice + 1)), etaWidth));
  }

  return slices;
}

}  // namespace EtaSlicer

}  // namespace ActsExamples
