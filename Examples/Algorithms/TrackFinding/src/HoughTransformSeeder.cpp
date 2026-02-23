// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/HoughTransformSeeder.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/MathHelpers.hpp"
#include "ActsExamples/EventData/GeometryContainers.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/TrackFinding/DefaultHoughFunctions.hpp"
#include "ActsExamples/Utilities/GroupBy.hpp"
#include "ActsFatras/EventData/Barcode.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iterator>
#include <ostream>
#include <stdexcept>

static inline int quant(double min, double max, unsigned nSteps, double val);
static inline double unquant(double min, double max, unsigned nSteps, int step);
static inline double unquantSteps(double previous, double stepSize,
                                  unsigned nSteps, unsigned int from,
                                  unsigned iStep);
static inline double unquantEqudistantPt(double min, double max,
                                         unsigned nSteps, int step,
                                         const std::vector<double>& ptBins);
static inline double unquantFinerCentral(double previous, double stepSize,
                                         unsigned nSteps, unsigned int from,
                                         double factor, unsigned iStep);
template <typename T>
static inline std::string to_string(std::vector<T> v);

thread_local std::vector<std::shared_ptr<ActsExamples::HoughMeasurementStruct>>
    houghMeasurementStructs;
thread_local std::unordered_set<int> populatedLayers;

ActsExamples::HoughTransformSeeder::HoughTransformSeeder(
    ActsExamples::HoughTransformSeeder::Config cfg, Acts::Logging::Level lvl)
    : ActsExamples::IAlgorithm("HoughTransformSeeder", lvl),
      m_cfg(std::move(cfg)),
      m_logger(Acts::getDefaultLogger("HoughTransformSeeder", lvl)),
      m_writer(std::make_unique<Writer>(m_cfg.writeToSingleFile)) {
  // require spacepoints or input measurements (or both), but at least one kind
  // of input
  bool foundInput = false;
  for (const auto& spName : m_cfg.inputSpacePoints) {
    if (!(spName.empty())) {
      foundInput = true;
    }

    auto& handle = m_inputSpacePoints.emplace_back(
        std::make_unique<ReadDataHandle<SimSpacePointContainer>>(
            this,
            "InputSpacePoints#" + std::to_string(m_inputSpacePoints.size())));
    handle->initialize(spName);
  }
  if (!(m_cfg.inputMeasurements.empty())) {
    foundInput = true;
  }

  if (!foundInput) {
    throw std::invalid_argument(
        "HoughTransformSeeder: Missing some kind of input (measurements of "
        "spacepoints)");
  }

  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument(
        "HoughTransformSeeder: Missing hough tracks output collection");
  }
  if (m_cfg.outputSeeds.empty()) {
    throw std::invalid_argument(
        "HoughTransformSeeder: Missing hough track seeds output collection");
  }

  m_outputProtoTracks.initialize(m_cfg.outputProtoTracks);
  m_inputMeasurements.initialize(m_cfg.inputMeasurements);
  m_inputMeasurementParticlesMap.initialize("measurement_particles_map");
  m_inputParticles.initialize("particles_simulated");

  if (!m_cfg.trackingGeometry) {
    throw std::invalid_argument(
        "HoughTransformSeeder: Missing tracking geometry");
  }

  if (m_cfg.geometrySelection.empty()) {
    throw std::invalid_argument(
        "HoughTransformSeeder: Missing geometry selection");
  }

  auto hasInvalidInputs = [](const Acts::GeometryIdentifier& geoId) {
    return (geoId.approach() != 0u) || (geoId.boundary() != 0u) ||
           (geoId.sensitive() != 0u);
  };

  if (std::ranges::any_of(m_cfg.geometrySelection, hasInvalidInputs)) {
    throw std::invalid_argument(
        "HoughTransformSeeder: Invalid geometry selection: only volume and "
        "layer are allowed to be set");
  }

  // remove geometry selection duplicates
  //
  // the geometry selections must be mutually exclusive, i.e. if we have a
  // selection that contains both a volume and a layer within that same volume,
  // we would create the space points for the layer twice.
  auto isDuplicate = [](Acts::GeometryIdentifier ref,
                        Acts::GeometryIdentifier cmp) {
    // code assumes ref < cmp and that only volume and layer can be non-zero
    // root node always contains everything
    if (ref.volume() == 0) {
      return true;
    }
    // unequal volumes always means separate hierarchies
    if (ref.volume() != cmp.volume()) {
      return false;
    }
    // within the same volume hierarchy only consider layers
    return (ref.layer() == cmp.layer());
  };
  // sort geometry selection so the unique filtering works
  std::ranges::sort(m_cfg.geometrySelection,
                    std::less<Acts::GeometryIdentifier>{});
  auto geoSelBeg = m_cfg.geometrySelection.begin();
  auto geoSelEnd = m_cfg.geometrySelection.end();
  auto geoSelLastUnique = std::unique(geoSelBeg, geoSelEnd, isDuplicate);
  if (geoSelLastUnique != geoSelEnd) {
    ACTS_WARNING("Removed " << std::distance(geoSelLastUnique, geoSelEnd)
                            << " geometry selection duplicates");
    m_cfg.geometrySelection.erase(geoSelLastUnique, geoSelEnd);
  }
  ACTS_INFO("Hough geometry selection:");
  for (const auto& geoId : m_cfg.geometrySelection) {
    ACTS_INFO("  " << geoId);
  }

  std::vector<double> ptBins;
  const double ptBinSize = 1.;  // GeV
  const double minPt = 1.;
  const double maxPt = m_cfg.houghHistSize_y * ptBinSize;
  const unsigned halfSize = m_cfg.houghHistSize_y / 2;
  for (unsigned i = 0; i < halfSize; ++i) {
    ptBins.push_back(minPt +
                     (maxPt - minPt) * i / static_cast<double>(halfSize));
  }

  // Fill convenience variables
  m_step_x = (m_cfg.xMax - m_cfg.xMin) / m_cfg.houghHistSize_x;
  m_step_y = (m_cfg.yMax - m_cfg.yMin) / m_cfg.houghHistSize_y;
  for (unsigned i = 0; i <= m_cfg.houghHistSize_x; i++) {
    m_bins_x.push_back(
        unquant(m_cfg.xMin, m_cfg.xMax, m_cfg.houghHistSize_x, i));
  }

  const int fineBins = 50;
  const float fineFactor = 4.;
  if (m_cfg.binning == Binning::FinerCentral) {
    m_step_y = (m_cfg.yMax - m_cfg.yMin) / (fineBins + fineFactor * (m_cfg.houghHistSize_y - fineBins));
  }

  for (unsigned i = 0; i <= m_cfg.houghHistSize_y; i++) {
    if (m_cfg.binning == Binning::EqudistantQoverPt) {
      m_bins_y.push_back(
          unquant(m_cfg.yMin, m_cfg.yMax, m_cfg.houghHistSize_y, i));
    } else if (m_cfg.binning == Binning::EqudistantPt) {
      m_bins_y.push_back(unquantEqudistantPt(m_cfg.yMin, m_cfg.yMax,
                                             m_cfg.houghHistSize_y, i, ptBins));
    } else if (m_cfg.binning == Binning::Steps) {
      m_bins_y.push_back(unquantSteps(m_bins_y.back(), m_step_y,
                                      m_cfg.houghHistSize_y, 54, i));
    } else if (m_cfg.binning == Binning::FinerCentral) {
      m_bins_y.push_back(unquantFinerCentral(m_bins_y.back(), m_step_y,
                                             m_cfg.houghHistSize_y, fineBins, fineFactor, i));
    }
  }

  m_cfg.fieldCorrector
      .connect<&ActsExamples::DefaultHoughFunctions::fieldCorrectionDefault>();
  m_cfg.layerIDFinder
      .connect<&ActsExamples::DefaultHoughFunctions::findLayerIDDefault>();

  auto slicerEquidistantEta =
      [](const std::shared_ptr<HoughMeasurementStruct>& meas,
         int slice) -> ResultBool {
    if (slice == -1) {
      return ResultBool::success(true);
    }

    auto easing = [](double x) {
      // return ((0 < x) - (x < 0)) * 32 *
      // (1 - std::cos((x * std::numbers::pi) / 64));  // InSine
      return ((0 < x) - (x < 0)) * 11 * (x * x / 121.);  // InSquare
      // return 32 * (x * x * x / 32768);  // InCubic
      // return ((0 < x) - (x < 0)) * (32 - std::sqrt(1024 - x * x));  // InCirc
      // return x;  // Linear
    };

    const double lo_cot = easing(-11.0 + 11. / 16 * slice);
    const double hi_cot = easing(-11.0 + 11. / 16. * (slice + 1));
    const double v1 = (meas->z + 200) / meas->radius;
    const double v2 = (meas->z - 200) / meas->radius;

    return ResultBool::success((v1 - lo_cot) * (v2 - hi_cot) < 0);
  };

  auto slicerNone = [](const std::shared_ptr<HoughMeasurementStruct>&,
                       int slice) -> ResultBool {
    return ResultBool::success(slice == -1);
  };

  auto slicerWedges = [](const std::shared_ptr<HoughMeasurementStruct>& meas,
                         int slice) -> ResultBool {
    if (slice == -1) {
      return ResultBool::success(true);
    }

    return ResultBool::success(
        Wedges::wedges[slice].in_rPhiZ(meas->radius, meas->phi, meas->z));
  };

  switch (m_cfg.slicing) {
    case ActsExamples::Slicing::Wedges:
      m_cfg.sliceTester.connect<slicerWedges>();
      break;
    case ActsExamples::Slicing::EqudistantEta:
      m_cfg.sliceTester.connect<slicerEquidistantEta>();
      break;
    case ActsExamples::Slicing::None:
      m_cfg.sliceTester.connect<slicerNone>();
    default:
      break;
  }
}

ActsExamples::ProcessCode ActsExamples::HoughTransformSeeder::execute(
    const AlgorithmContext& ctx) const {
  // clear our Hough measurements out from the previous iteration, if at all
  houghMeasurementStructs.clear();
  populatedLayers.clear();

  // add SPs to the inputs
  addSpacePoints(ctx);

  // add ACTS measurements
  addMeasurements(ctx);

  const auto& measurementParticleMap = m_inputMeasurementParticlesMap(ctx);
  const auto& particles = m_inputParticles(ctx);

  static thread_local ProtoTrackContainer protoTracks;
  protoTracks.clear();

  // loop over our subregions and run the Hough Transform on each
  for (int subregion : m_cfg.subRegions) {
    ACTS_DEBUG("Processing subregion " << subregion);
    ActsExamples::HoughHist m_houghHist = createHoughHist(subregion);

    const auto hough_name =
        std::format("event_{:06}_{:02}", ctx.eventNumber, subregion);
    const auto hough_title =
        std::format("event_{:06}_{:02};q/p_{{T}} [1/GeV];#varphi [rad]",
                    ctx.eventNumber, subregion);
    auto hough_hist = std::unique_ptr<TH2S>(
        new TH2S(hough_name.c_str(), hough_title.c_str(), m_cfg.houghHistSize_y,
                 m_bins_y.data(), m_cfg.houghHistSize_x, m_bins_x.data()));

    for (unsigned y = 0; y < m_cfg.houghHistSize_y; y++) {
      for (unsigned x = 0; x < m_cfg.houghHistSize_x; x++) {
        if (unsigned entries = m_houghHist.nLayers(y, x); entries > 0) {
          ACTS_DEBUG(std::format("bin (q/pT, phi) = ({}, {})", y, x));
          // Flat layers
          // hh_hist->SetBinContent(y + 1, x + 1, entries);

          // Bit pattern
          const std::uint16_t bits = std::accumulate(
              m_houghHist.layers(y, x).begin(), m_houghHist.layers(y, x).end(),
              std::uint16_t{}, [](std::uint16_t sum, std::uint16_t layer) {
                return sum | 0x1 << layer;
              });
          hough_hist->SetBinContent(y + 1, x + 1, bits);
          ACTS_DEBUG(std::format("bitmask={} n_bits={}",
                                 std::bitset<16>(bits).to_string(), entries));

          if (entries < m_cfg.truthThreshold) {
            continue;
          }

          // Find truth particle contributing the most
          std::vector<std::uint64_t> particle_hashes;
          for (const HoughMeasurement index : m_houghHist.hitIds(y, x)) {
            for (const Index measurement_index :
                 houghMeasurementStructs[index]->indices) {
              particle_hashes.push_back(
                  measurementParticleMap.find(measurement_index)
                      ->second.hash());
            }
          }
          ACTS_DEBUG(std::format("n_measurements={}", particle_hashes.size()));

          std::map<std::uint64_t, std::uint32_t> counts;
          for (std::uint64_t barcode : particle_hashes) {
            counts[barcode]++;
          }

          if (logger().doPrint(Acts::Logging::DEBUG)) {
            for (const auto& [hash, count] : counts) {
              ACTS_DEBUG(std::format("\t{} -> {}", hash, count));
            }
          }

          const auto& [hash, count] = *std::max_element(
              counts.begin(), counts.end(), [](const auto lhs, const auto rhs) {
                return lhs.second < rhs.second;
              });

          if (count * 2 >= particle_hashes.size()) {
            const auto particle =
                std::find_if(particles.begin(), particles.end(),
                             [hash](const SimParticle& p) {
                               return p.particleId().hash() == hash;
                             });
            if (particle != particles.end() &&
                particle->transverseMomentum() > 1.) {
              ACTS_DEBUG(std::format("particle={} pt={}", hash,
                                     particle->transverseMomentum()));
              m_writer->writeTree(ctx.eventNumber, subregion, y, x, hash,
                                  count);
            }
          }
        }

        if (!passThreshold(m_houghHist, x, y)) {
          continue;
        }

        // FIXME: Disabling writing to containers temporarily to avoid memory
        // issues when generating a ttbar sample with very high pile-up
        continue;

        // Now we need to unpack the hits; there should be multiple track
        // candidates if we have multiple hits in a given layer. So the first
        // thing is to unpack the indices (which is what we need) by layer

        std::vector<std::vector<std::vector<Index>>> hitIndicesAll(
            m_cfg.nLayers);
        std::vector<std::size_t> nHitsPerLayer(m_cfg.nLayers);
        for (auto measurementIndex : m_houghHist.hitIds(y, x)) {
          HoughMeasurementStruct* meas =
              houghMeasurementStructs[measurementIndex].get();
          hitIndicesAll[meas->layer].push_back(meas->indices);
          nHitsPerLayer[meas->layer]++;
        }

        std::vector<std::vector<int>> combs = getComboIndices(nHitsPerLayer);

        // Loop over all combinations.
        for (auto [icomb, hit_indices] : Acts::enumerate(combs)) {
          ProtoTrack protoTrack;
          for (unsigned layer = 0; layer < m_cfg.nLayers; layer++) {
            if (hit_indices[layer] >= 0) {
              for (auto index : hitIndicesAll[layer][hit_indices[layer]]) {
                protoTrack.push_back(index);
              }
            }
          }
          protoTracks.push_back(protoTrack);
        }
      }
    }

    // Sliding window
    const auto peaks_name =
        std::format("peaks_{:06}_{:02}", ctx.eventNumber, subregion);
    const auto peaks_title =
        std::format("peaks_{:06}_{:02};q/p_{{T}} [1/GeV];#varphi [rad]",
                    ctx.eventNumber, subregion);
    auto peaks_hist = std::unique_ptr<TH2S>(
        new TH2S(peaks_name.c_str(), peaks_title.c_str(), m_cfg.houghHistSize_y,
                 m_bins_y.data(), m_cfg.houghHistSize_x, m_bins_x.data()));

    const auto all_peaks = slidingWindowPeaks(m_houghHist, m_cfg.slidingWindow);
    ACTS_DEBUG(std::format("Found {} peaks", all_peaks.size()));
    for (const auto& peak : all_peaks) {
      ACTS_DEBUG(std::format("peak=({},{}) bin=({},{})", m_bins_y[peak[0]],
                             m_bins_x[peak[1]], peak[0] + 1, peak[1] + 1));
      peaks_hist->Fill(m_bins_y[peak[0]], m_bins_x[peak[1]]);
    }

    if (m_cfg.writeToSingleFile) {
      m_writer->writeObj(hough_hist.get());
      m_writer->writeObj(peaks_hist.get());
    } else {
      m_writer->writeObjThread(hough_hist.get());
      m_writer->writeObjThread(peaks_hist.get());
    }
  }
  ACTS_DEBUG("Created " << protoTracks.size() << " proto track");

  m_outputProtoTracks(ctx, ProtoTrackContainer{protoTracks});
  // clear the vector
  houghMeasurementStructs.clear();
  return ActsExamples::ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode ActsExamples::HoughTransformSeeder::finalize() {
  m_writer->close();

  return ActsExamples::ProcessCode::SUCCESS;
}

ActsExamples::HoughHist ActsExamples::HoughTransformSeeder::createHoughHist(
    int subregion) const {
  ActsExamples::HoughHist houghHist(m_cfg.plane);

  for (unsigned int layer : populatedLayers) {
    auto filter_layer_slice =
        [layer, subregion,
         this](const std::shared_ptr<HoughMeasurementStruct>& meas) {
          return meas->layer == layer &&
                 m_cfg.sliceTester(meas, subregion).value();
        };

    for (const auto& meas :
         houghMeasurementStructs | std::views::filter(filter_layer_slice)) {
      const std::uint32_t index =
          std::distance(houghMeasurementStructs.begin(),
                        std::find(houghMeasurementStructs.begin(),
                                  houghMeasurementStructs.end(), meas));

      for (unsigned y_ = 0; y_ < m_cfg.houghHistSize_y; y_++) {
        const unsigned y_bin_min = y_;
        const unsigned y_bin_max = (y_ + 1);

        // Find the min/max x bins
        const auto xBins = yToXBins(y_bin_min, y_bin_max, meas->radius,
                                    meas->phi, meas->layer);
        // Update the houghHist
        for (unsigned y = y_bin_min; y < y_bin_max; y++) {
          // Handle cases
          const double diff = xBins.second - xBins.first;
          if (diff < m_cfg.houghHistSize_x / 2) {
            for (unsigned x = xBins.first; x < xBins.second; x++) {
              houghHist.fillBin(y, x, index, layer);
            }
          } else {
            for (unsigned x = 0; x < xBins.first; ++x) {
              houghHist.fillBin(y, x, index, layer);
            }
            for (unsigned x = xBins.second; x < m_cfg.houghHistSize_x; ++x) {
              houghHist.fillBin(y, x, index, layer);
            }
          }
        }
      }
    }
  }

  return houghHist;
}

bool ActsExamples::HoughTransformSeeder::passThreshold(
    HoughHist const& houghHist, unsigned x, unsigned y) const {
  // Pass window threshold
  unsigned width = m_cfg.threshold.size() / 2;
  if (x < width || m_cfg.houghHistSize_x - x < width) {
    return false;
  }
  for (unsigned i = 0; i < m_cfg.threshold.size(); i++) {
    if (houghHist.nLayers(y, x - width + i) < m_cfg.threshold[i]) {
      return false;
    }
  }

  // Pass local-maximum check, if used
  if (m_cfg.localMaxWindowSize != 0) {
    for (int j = -m_cfg.localMaxWindowSize; j <= m_cfg.localMaxWindowSize;
         j++) {
      for (int i = -m_cfg.localMaxWindowSize; i <= m_cfg.localMaxWindowSize;
           i++) {
        if (i == 0 && j == 0) {
          continue;
        }
        if (y + j < m_cfg.houghHistSize_y && x + i < m_cfg.houghHistSize_x) {
          if (houghHist.nLayers(y + j, x + i) > houghHist.nLayers(y, x)) {
            return false;
          }
          if (houghHist.nLayers(y + j, x + i) == houghHist.nLayers(y, x)) {
            if (houghHist.nHits(y + j, x + i) > houghHist.nHits(y, x)) {
              return false;
            }
            if (houghHist.nHits(y + j, x + i) == houghHist.nHits(y, x) &&
                j <= 0 && i <= 0) {
              return false;  // favor bottom-left (low phi, low neg q/pt)
            }
          }
        }
      }
    }
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Helpers

// Quantizes val, given a range [min, max) split into nSteps. Returns the bin
// below.
static inline int quant(double min, double max, unsigned nSteps, double val) {
  return static_cast<int>((val - min) / (max - min) * nSteps);
}

// Returns the lower bound of the bin specified by step
static inline double unquant(double min, double max, unsigned nSteps,
                             int step) {
  return min + (max - min) * step / nSteps;
}

static inline double unquantSteps(double previous, double stepSize,
                                  unsigned nSteps, unsigned from,
                                  unsigned iStep) {
  if (iStep == 0) {
    return -1;
  }

  const unsigned half = nSteps / 2;
  const float factor = 2;
  if (iStep <= from / 2 || iStep > nSteps - from / 2) {
    return previous + stepSize * factor;
  } else if (iStep <= half - factor * from / 2 ||
             iStep > nSteps - half + factor * from / 2) {
    return previous + stepSize;
  } else {
    return previous + stepSize / factor;
  }
}

static inline double unquantFinerCentral(double previous, double stepSize,
                                         unsigned nSteps, unsigned from,
                                         double factor, unsigned iStep) {
  if (iStep == 0) {
    return -1;
  }

  const unsigned half = nSteps / 2;
  if (iStep <= half -  from / 2 || iStep > nSteps - half + from / 2) {
    return previous + factor * stepSize;
  } else {
    return previous + stepSize;
  }
}

// Returns the lower bound of the bin specified by step
static inline double unquantEqudistantPt(double min, double max,
                                         unsigned nSteps, int step,
                                         const std::vector<double>& ptBins) {
  if (const double qOverp = unquant(min, max, nSteps, step); qOverp < 0) {
    const double ptBin = ptBins[step];
    return -1. / ptBin;
  } else if (qOverp == 0.) {
    return 0;
  } else {
    const double ptBin = ptBins[nSteps - step];
    return 1. / ptBin;
  }
}

template <typename T>
static inline std::string to_string(std::vector<T> v) {
  std::ostringstream oss;
  oss << "[";
  if (!v.empty()) {
    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(oss, ", "));
    oss << v.back();
  }
  oss << "]";
  return oss.str();
}

double ActsExamples::HoughTransformSeeder::yToX(double y, double r,
                                                double phi) const {
  double d0 = 0;  // d0 correction TO DO allow for this
  double x =
      std::asin(r * ActsExamples::HoughTransformSeeder::m_cfg.kA * y - d0 / r) +
      phi;

  if (m_cfg.fieldCorrector.connected()) {
    x += (m_cfg.fieldCorrector(0, y, r)).value();
  }

  return std::remainder(x, 2.0 * std::numbers::pi);
}

// Find the min/max x bins of the hit's line, in each y bin. Max is exclusive.
// Note this assumes yToX is monotonic. Returns {0, 0} if hit lies out of
// bounds.
std::pair<unsigned, unsigned> ActsExamples::HoughTransformSeeder::yToXBins(
    std::size_t yBin_min, std::size_t yBin_max, double r, double phi,
    unsigned layer) const {
  double x_min = yToX(m_bins_y[yBin_min], r, phi);
  double x_max = yToX(m_bins_y[yBin_max], r, phi);
  if (x_min > x_max) {
    std::swap(x_min, x_max);
  }
  if (x_max < m_cfg.xMin || x_min > m_cfg.xMax) {
    return {0, 0};  // out of bounds
  }

  // Get bins
  int x_bin_min = quant(m_cfg.xMin, m_cfg.xMax, m_cfg.houghHistSize_x, x_min);
  int x_bin_max = quant(m_cfg.xMin, m_cfg.xMax, m_cfg.houghHistSize_x, x_max) +
                  1;  // exclusive

  // Extend bins
  unsigned extend = getExtension(yBin_min, layer);
  x_bin_min -= extend;
  x_bin_max += extend;

  // Clamp bins
  if (x_bin_min < 0) {
    x_bin_min = 0;
  }
  if (x_bin_max > static_cast<int>(m_cfg.houghHistSize_x)) {
    x_bin_max = m_cfg.houghHistSize_x;
  }

  return {x_bin_min, x_bin_max};
}

// We allow variable extension based on the size of m_hitExtend_x. See
// comments below.
unsigned ActsExamples::HoughTransformSeeder::getExtension(
    unsigned y, unsigned layer) const {
  if (m_cfg.hitExtend_x.size() == m_cfg.nLayers) {
    return m_cfg.hitExtend_x[layer];
  }

  if (m_cfg.hitExtend_x.size() == m_cfg.nLayers * 2) {
    // different extension for low pt vs high pt, split in half but
    // irrespective of sign first nLayers entries of m_hitExtend_x is for low
    // pt half, rest are for high pt half
    if (y < m_cfg.houghHistSize_y / 4 || y > 3 * m_cfg.houghHistSize_y / 4) {
      return m_cfg.hitExtend_x[layer];
    }

    return m_cfg.hitExtend_x[m_cfg.nLayers + layer];
  }
  return 0;
}

/**
 * Given a list of sizes (of arrays), generates a list of all combinations of
 * indices to index one element from each array.
 *
 * For example, given [2 3], generates [(0 0) (1 0) (0 1) (1 1) (0 2) (1 2)].
 *
 * This basically amounts to a positional number system of where each digit
 * has its own base. The number of digits is sizes.size(), and the base of
 * digit i is sizes[i]. Then all combinations can be uniquely represented just
 * by counting from [0, nCombs).
 *
 * For a decimal number like 1357, you get the thousands digit with n / 1000 =
 * n / (10 * 10 * 10). So here, you get the 0th digit with n / (base_1 *
 * base_2 * base_3);
 */
std::vector<std::vector<int>>
ActsExamples::HoughTransformSeeder::getComboIndices(
    std::vector<std::size_t>& sizes) const {
  std::size_t nCombs = 1;
  std::vector<std::size_t> nCombs_prior(sizes.size());
  std::vector<int> temp(sizes.size(), 0);

  for (std::size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] > 0) {
      nCombs_prior[i] = nCombs;
      nCombs *= sizes[i];
    } else {
      temp[i] = -1;
    }
  }

  std::vector<std::vector<int>> combos(nCombs, temp);

  for (std::size_t icomb = 0; icomb < nCombs; icomb++) {
    std::size_t index = icomb;
    for (std::size_t isize = sizes.size() - 1; isize < sizes.size(); isize--) {
      if (sizes[isize] == 0) {
        continue;
      }
      combos[icomb][isize] = static_cast<int>(index / nCombs_prior[isize]);
      index = index % nCombs_prior[isize];
    }
  }

  return combos;
}

void ActsExamples::HoughTransformSeeder::addSpacePoints(
    const AlgorithmContext& ctx) const {
  // construct the combined input container of space point pointers from all
  // configured input sources.

  // auto file = TFile::Open("hitmaps.root", "recreate");
  // std::unordered_map<int, TH2F> zr, xy;
  // for (int slice : m_cfg.subRegions) {
  //   {
  //     const auto name = (slice == -1) ? "zr_all" : std::format("zr_{}",
  //     slice); zr[slice] = {name.c_str(), name.c_str(), 800, -3200, 3200, 400,
  //     0, 1200};
  //   }
  //   {
  //     const auto name = (slice == -1) ? "xy_all" : std::format("xy_{}",
  //     slice); xy[slice] = {name.c_str(), name.c_str(), 400,   -1200,
  //                  1200,         400,          -1200, 1200};
  //   }
  // }
  for (const auto& isp : m_inputSpacePoints) {
    const auto& spContainer = (*isp)(ctx);
    ACTS_DEBUG("Inserting " << spContainer.size() << " space points from "
                            << isp->key());
    for (auto& sp : spContainer) {
      const double r = Acts::fastHypot(sp.x(), sp.y());
      const double z = sp.z();
      const float phi = std::atan2(sp.y(), sp.x());
      const double theta = std::atan2(r, z);
      const double eta = -std::log(std::tan(theta / 2.));
      const ResultUnsigned hitlayer = m_cfg.layerIDFinder(r).value();
      if (!(hitlayer.ok())) {
        continue;
      }
      ACTS_DEBUG(std::format("{}: r={} z={} layer={}",
                             r < 350 ? "PIXEL" : "STRIP", r, z,
                             hitlayer.value()));
      std::vector<Index> indices;
      for (const auto& slink : sp.sourceLinks()) {
        const auto& islink = slink.get<IndexSourceLink>();
        indices.push_back(islink.index());
      }

      populatedLayers.insert(hitlayer.value());

      auto meas =
          std::shared_ptr<HoughMeasurementStruct>(new HoughMeasurementStruct(
              hitlayer.value(), phi, r, z, eta, indices, HoughHitType::SP));
      houghMeasurementStructs.push_back(meas);
      // for (int slice : m_cfg.subRegions) {
      //   if (m_cfg.sliceTester(meas, slice).value()) {
      //     zr[slice].Fill(z, r);
      //     if ((r < 200 && std::fabs(z) < 600) ||
      //         (r > 200 && std::fabs(z) < 1200)) {
      //       xy[slice].Fill(sp.x(), sp.y());
      //     }
      //   }
      // }
    }
  }
  // file->Write();
  // file->Close();
}

void ActsExamples::HoughTransformSeeder::addMeasurements(
    const AlgorithmContext& ctx) const {
  const auto& measurements = m_inputMeasurements(ctx);

  ACTS_DEBUG("Inserting " << measurements.size() << " space points from "
                          << m_cfg.inputMeasurements);

  for (Acts::GeometryIdentifier geoId : m_cfg.geometrySelection) {
    // select volume/layer depending on what is set in the geometry id
    auto range =
        selectLowestNonZeroGeometryObject(measurements.orderedIndices(), geoId);
    // groupByModule only works with geometry containers, not with an
    // arbitrary range. do the equivalent grouping manually
    auto groupedByModule = makeGroupBy(range, detail::GeometryIdGetter());

    for (const auto& [moduleGeoId, moduleSourceLinks] : groupedByModule) {
      // find corresponding surface
      const Acts::Surface* surface =
          m_cfg.trackingGeometry->findSurface(moduleGeoId);
      if (surface == nullptr) {
        ACTS_ERROR("Could not find surface " << moduleGeoId);
        return;
      }

      for (auto& sourceLink : moduleSourceLinks) {
        // extract a local position/covariance independent of the concrete
        // measurement content. since we do not know if and where the local
        // parameters are contained in the measurement parameters vector, they
        // are transformed to the bound space where we do know their location.
        // if the local parameters are not measured, this results in a
        // zero location, which is a reasonable default fall-back.
        const ConstVariableBoundMeasurementProxy measurement =
            measurements.getMeasurement(sourceLink.index());

        assert(measurement.contains(Acts::eBoundLoc0) &&
               "Measurement does not contain the required bound loc0");
        assert(measurement.contains(Acts::eBoundLoc1) &&
               "Measurement does not contain the required bound loc1");

        auto boundLoc0 = measurement.indexOf(Acts::eBoundLoc0);
        auto boundLoc1 = measurement.indexOf(Acts::eBoundLoc1);

        Acts::Vector2 localPos{measurement.parameters()[boundLoc0],
                               measurement.parameters()[boundLoc1]};

        // transform local position to global coordinates
        Acts::Vector3 globalFakeMom(1, 1, 1);
        Acts::Vector3 globalPos =
            surface->localToGlobal(ctx.geoContext, localPos, globalFakeMom);
        const double r = globalPos.head<2>().norm();
        const double phi =
            std::atan2(globalPos[Acts::ePos1], globalPos[Acts::ePos0]);
        const double z = globalPos[Acts::ePos2];
        const double theta = std::atan2(r, z);
        const double eta = -std::log(std::tan(theta / 2.));
        const ResultUnsigned hitlayer = m_cfg.layerIDFinder(r);
        if (hitlayer.ok()) {
          std::vector<Index> index;
          index.push_back(sourceLink.index());
          populatedLayers.insert(hitlayer.value());
          auto houghMeas = std::shared_ptr<HoughMeasurementStruct>(
              new HoughMeasurementStruct(hitlayer.value(), phi, r, z, eta,
                                         index, HoughHitType::MEASUREMENT));
          houghMeasurementStructs.push_back(houghMeas);
        }
      }
    }
  }
}
