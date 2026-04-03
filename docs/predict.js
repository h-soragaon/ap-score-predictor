/**
 * AP Score Predictor — Client-side prediction engine.
 *
 * Ports the Python ordered-logit pipeline to JavaScript so the entire
 * app can run as a static site (GitHub Pages) with zero backend.
 */

const Predictor = (function () {
  'use strict';

  const DIFFICULTY_LAMBDA = 1.0;
  const SHRINK_FACTOR = 0.7;

  // ── Logistic (sigmoid) function ────────────────

  function expit(x) {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
  }

  // ── Ordered logit: P(score = 1..5) ────────────

  function orderedLogitProbs(x, tau, sigma) {
    const cumulative = tau.map(t => expit((t - x) / sigma));
    const probs = new Array(5);
    probs[0] = cumulative[0];
    for (let k = 1; k < 4; k++) {
      probs[k] = cumulative[k] - cumulative[k - 1];
    }
    probs[4] = 1.0 - cumulative[3];

    // Clip tiny negatives from numerical issues
    for (let i = 0; i < 5; i++) {
      probs[i] = Math.max(0, Math.min(1, probs[i]));
    }
    const sum = probs.reduce((a, b) => a + b, 0);
    for (let i = 0; i < 5; i++) {
      probs[i] /= sum;
    }
    return probs;
  }

  // ── Weighted composite score ───────────────────

  function computeWeightedComposite(mcqCorrect, frqSections, config) {
    let composite = (mcqCorrect / config.mcq_total) * config.mcq_weight;

    for (const sectionConfig of config.frq_sections) {
      const sectionInput = frqSections[sectionConfig.name];
      for (let i = 0; i < sectionConfig.question_max.length; i++) {
        const score = sectionInput[i];
        const maxPts = sectionConfig.question_max[i];
        const qWeight = sectionConfig.question_weights[i];
        const proportion = maxPts > 0 ? score / maxPts : 0;
        composite += proportion * qWeight;
      }
    }

    return Math.max(0, Math.min(1, composite));
  }

  // ── Difficulty adjustment ──────────────────────

  function computeDifficultyAdjustment(frqSections, config, stats) {
    if (!stats) return 0;

    // Index stats by "section_question"
    const statsMap = {};
    for (const q of stats) {
      statsMap[q.section + '_' + q.question] = q;
    }

    let weightedZSum = 0;
    let totalWeight = 0;

    for (const sectionConfig of config.frq_sections) {
      const sectionInput = frqSections[sectionConfig.name];
      for (let i = 0; i < sectionConfig.question_max.length; i++) {
        const key = sectionConfig.name + '_' + (i + 1);
        const stat = statsMap[key];
        if (!stat) continue;
        if (stat.sd <= 0 || sectionConfig.question_max[i] <= 0) continue;

        const maxPts = sectionConfig.question_max[i];
        const qWeight = sectionConfig.question_weights[i];
        const pStudent = sectionInput[i] / maxPts;
        const pNational = stat.mean / stat.max_points;
        const sdProportion = stat.sd / stat.max_points;

        const z = (pStudent - pNational) / sdProportion;
        // Shrink extremes: tanh-based soft clipping
        const zShrunk = Math.tanh(z * SHRINK_FACTOR) / SHRINK_FACTOR;

        weightedZSum += zShrunk * qWeight;
        totalWeight += qWeight;
      }
    }

    if (totalWeight === 0) return 0;
    const rawAdj = weightedZSum / totalWeight;
    return rawAdj * 0.03;
  }

  // ── Explanations ───────────────────────────────

  function generateExplanations(composite, diffAdj, probs, mostLikely) {
    const explanations = [];

    if (composite >= 0.75) {
      explanations.push('Strong overall performance across both MCQ and FRQ sections.');
    } else if (composite >= 0.5) {
      explanations.push('Moderate overall performance; composite is near the course median.');
    } else {
      explanations.push('Below-average composite score relative to typical performance ranges.');
    }

    if (Math.abs(diffAdj) > 0.005) {
      if (diffAdj > 0) {
        explanations.push(
          'FRQ performance was above the national means on available questions, ' +
          'providing a positive adjustment.'
        );
      } else {
        explanations.push(
          'FRQ performance was below the national means on available questions, ' +
          'providing a negative adjustment.'
        );
      }
    } else {
      explanations.push(
        'No significant difficulty adjustment (no scoring statistics available ' +
        'or performance was near national averages).'
      );
    }

    const maxProb = Math.max(...probs);
    if (maxProb < 0.4) {
      explanations.push(
        `Result is near a score boundary \u2014 uncertainty is high. ` +
        `The most likely score of ${mostLikely} has only ${Math.round(maxProb * 100)}% probability.`
      );
    } else if (maxProb > 0.7) {
      explanations.push(
        `The prediction is fairly confident: score ${mostLikely} ` +
        `has ${Math.round(maxProb * 100)}% probability.`
      );
    }

    return explanations;
  }

  // ── Main predict function ──────────────────────

  function predict(input, config, priors, scoringStats) {
    // 1. Composite
    const composite = computeWeightedComposite(input.mcqCorrect, input.frqSections, config);

    // 2. Difficulty adjustment
    const diffAdj = computeDifficultyAdjustment(input.frqSections, config, scoringStats);

    // 3. Adjusted composite
    const xAdj = Math.max(0, Math.min(1, composite + DIFFICULTY_LAMBDA * diffAdj));

    // 4. Ordered logit probabilities
    const probs = orderedLogitProbs(xAdj, priors.tau, priors.sigma);

    // 5. Build distribution
    const distribution = {};
    for (let k = 0; k < 5; k++) {
      distribution[String(k + 1)] = Math.round(probs[k] * 10000) / 10000;
    }

    // 6. Most likely and expected score
    let mostLikely = 1;
    let maxP = probs[0];
    for (let k = 1; k < 5; k++) {
      if (probs[k] > maxP) {
        maxP = probs[k];
        mostLikely = k + 1;
      }
    }

    let expected = 0;
    for (let k = 0; k < 5; k++) {
      expected += probs[k] * (k + 1);
    }

    // 7. Confidence band (p10, p90)
    const cumProbs = [];
    let cum = 0;
    for (let k = 0; k < 5; k++) {
      cum += probs[k];
      cumProbs.push(cum);
    }

    let p10 = 1;
    for (let k = 0; k < 5; k++) {
      if (cumProbs[k] >= 0.10) { p10 = k + 1; break; }
    }
    let p90 = 5;
    for (let k = 0; k < 5; k++) {
      if (cumProbs[k] >= 0.90) { p90 = k + 1; break; }
    }

    // 8. Explanations
    const explanations = generateExplanations(composite, diffAdj, probs, mostLikely);

    return {
      predicted_distribution: distribution,
      most_likely_score: mostLikely,
      expected_score: Math.round(expected * 100) / 100,
      weighted_composite: Math.round(composite * 10000) / 10000,
      difficulty_adjustment: Math.round(diffAdj * 10000) / 10000,
      confidence_band: { p10, p90 },
      explanations,
    };
  }

  return { predict };
})();
