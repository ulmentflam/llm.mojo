# ===----------------------------------------------------------------------=== #
# Unit tests for llmm.scheduler (LR curves). Run with `make test-mojo`.
#
# All tables use lr = 1.0 so the expected value is the multiplier itself; the
# expected numbers are worked out by hand in the comments.
# ===----------------------------------------------------------------------=== #

from std.testing import assert_almost_equal, assert_equal, TestSuite
from std.math import sqrt
from llmm.scheduler import LearningRateScheduler


def _lr(s: LearningRateScheduler, step: Int) raises -> Float32:
    return s.get_learning_rate(step)


def test_cosine_table() raises:
    # N=100, warmup 10, floor 0: warmup lr = (step+1)/10, then
    # 0.5*(1+cos(pi*(step-10)/90)).
    var s = LearningRateScheduler("cosine", 1.0, 10, 100, 0.0)
    assert_almost_equal(_lr(s, 0), 0.1, atol=1e-6)  # 1/10
    assert_almost_equal(_lr(s, 9), 1.0, atol=1e-6)  # 10/10
    assert_almost_equal(_lr(s, 10), 1.0, atol=1e-6)  # ratio 0
    assert_almost_equal(_lr(s, 55), 0.5, atol=1e-6)  # ratio 45/90 = 0.5
    # step 99: ratio 89/90, 0.5*(1+cos(89pi/90)) = 0.5*(1-cos(pi/90))
    assert_almost_equal(_lr(s, 99), 3.0459e-4, atol=1e-6)


def test_wsd_sqrt_default_table() raises:
    # N=100, warmup 10, floor 0, default cooldown 0.2 -> decay_point = 80.
    # decay: 1 - sqrt((step-80)/20).
    var s = LearningRateScheduler("wsd", 1.0, 10, 100, 0.0)
    assert_almost_equal(_lr(s, 0), 0.1, atol=1e-6)
    assert_almost_equal(_lr(s, 50), 1.0, atol=1e-6)
    assert_almost_equal(_lr(s, 79), 1.0, atol=1e-6)
    assert_almost_equal(_lr(s, 80), 1.0, atol=1e-6)  # ratio 0
    assert_almost_equal(_lr(s, 90), 0.292893, atol=1e-5)  # 1 - sqrt(0.5)
    assert_almost_equal(_lr(s, 99), 0.025321, atol=1e-5)  # 1 - sqrt(0.95)


def test_wsd_linear_cooldown_floor_table() raises:
    # N=100, warmup 0, cooldown 0.4 -> decay_point = 60, floor 0.1:
    # lr = 0.1 + (1 - r)*0.9, r = (step-60)/40.
    var s = LearningRateScheduler("wsd", 1.0, 0, 100, 0.1, 0.4, "linear")
    assert_almost_equal(_lr(s, 0), 1.0, atol=1e-6)  # no warmup, no div by 0
    assert_almost_equal(_lr(s, 59), 1.0, atol=1e-6)
    assert_almost_equal(_lr(s, 60), 1.0, atol=1e-6)
    assert_almost_equal(_lr(s, 80), 0.55, atol=1e-6)  # r=0.5 -> 0.1+0.45
    assert_almost_equal(_lr(s, 99), 0.1225, atol=1e-6)  # r=39/40 -> 0.1+0.0225


def test_wsd_sqrt_with_floor_table() raises:
    # Same cooldown, sqrt shape, floor 0.1, r=0.25 at step 70:
    # 0.1 + (1 - 0.5)*0.9 = 0.55.
    var s = LearningRateScheduler("wsd", 1.0, 0, 100, 0.1, 0.4, "sqrt")
    assert_almost_equal(_lr(s, 70), 0.55, atol=1e-6)


def test_zero_warmup_all_schedulers() raises:
    # -u 0 must give the full lr at step 0, no NaN/inf. (-k linear is left out
    # on purpose: it interpolates min_lr -> lr, i.e. it RISES; pre-existing,
    # not part of the WSD work, so its curve is not pinned here.)
    var c = LearningRateScheduler("cosine", 2.0, 0, 50, 0.0)
    var k = LearningRateScheduler("constant", 2.0, 0, 50, 0.0)
    var w = LearningRateScheduler("wsd", 2.0, 0, 50, 0.0)
    assert_almost_equal(_lr(c, 0), 2.0, atol=1e-6)
    assert_almost_equal(_lr(k, 0), 2.0, atol=1e-6)
    assert_almost_equal(_lr(w, 0), 2.0, atol=1e-6)


def test_cooldown_zero_is_flat() raises:
    # cooldown 0 -> decay_point == N: the divisor is clamped, curve stays flat.
    var s = LearningRateScheduler("wsd", 1.0, 0, 100, 0.0, 0.0, "linear")
    assert_almost_equal(_lr(s, 99), 1.0, atol=1e-6)


def test_wsd_defaults_match_original_formula() raises:
    # The original curve hard-coded decay_point = N*4//5 and 1 - sqrt(ratio).
    # Defaults must reproduce it bit for bit for every N and step.
    for n in range(1, 400):
        var s = LearningRateScheduler("wsd", 3e-4, 7, n, 0.05)
        var dp = n * 4 // 5
        for step in range(n):
            var expect: Float32
            if step < 7:
                expect = Float32(3e-4) * (Float32(step + 1) / Float32(7))
            elif step < dp:
                expect = 3e-4
            else:
                var r = Float32(step - dp) / Float32(n - dp)
                var mn = Float32(3e-4) * Float32(0.05)
                expect = mn + (1.0 - sqrt(r)) * (Float32(3e-4) - mn)
            assert_equal(_lr(s, step), expect)


def test_cooldown_one_decays_from_step_zero() raises:
    # cooldown 1.0 -> decay_point 0: linear from step 0, r = step/10, floor 0.
    var s = LearningRateScheduler("wsd", 1.0, 0, 10, 0.0, 1.0, "linear")
    assert_almost_equal(_lr(s, 0), 1.0, atol=1e-6)
    assert_almost_equal(_lr(s, 5), 0.5, atol=1e-6)
    assert_almost_equal(_lr(s, 9), 0.1, atol=1e-6)


def test_single_step_run() raises:
    # N=1: decay_point 0 under the default cooldown, ratio 0 -> full lr.
    var s = LearningRateScheduler("wsd", 1.0, 0, 1, 0.0)
    assert_almost_equal(_lr(s, 0), 1.0, atol=1e-6)
    var t = LearningRateScheduler("wsd", 1.0, 0, 1, 0.0, 0.4, "linear")
    assert_almost_equal(_lr(t, 0), 1.0, atol=1e-6)


def test_linear_matches_speedrun_form_at_ranking_configs() raises:
    # The W3 ranking runs (N=300, cooldown 0.4 and 0.45, floor 0.1) against
    # modded-nanogpt's multiplier, computed independently here in Float64:
    # w = min((1 - step/N) / cooldown, 1); mult = w + (1 - w) * floor.
    # N*cooldown is an integer for both, so the curves agree to f32 rounding
    # at every step; an off-by-one decay point moves some step by >= 1/120 of
    # the 0.9 range (~7.5e-3), a dropped floor or a sqrt shape by far more.
    var cds = [0.4, 0.45]
    for ci in range(2):
        var cd = cds[ci]
        var s = LearningRateScheduler(
            "wsd", 1.0, 0, 300, 0.1, Float32(cd), "linear"
        )
        for step in range(300):
            var w = min((1.0 - Float64(step) / 300.0) / cd, 1.0)
            var expect = w + (1.0 - w) * 0.1
            assert_almost_equal(Float64(_lr(s, step)), expect, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
