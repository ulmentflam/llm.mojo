from std.math import cos, pi, sqrt


struct LearningRateScheduler:
    var scheduler_type: StaticString
    var learning_rate: Scalar[DType.float32]
    var warmup_steps: Int
    var train_num_batches: Int
    var final_learning_rate_fraction: Scalar[DType.float32]
    # WSD only: fraction of training spent in the decay phase, and its shape
    # ("sqrt" = 1 - sqrt(r), the historical default; "linear" = 1 - r).
    var cooldown_fraction: Scalar[DType.float32]
    var decay_shape: StaticString

    def __init__(
        out self,
        scheduler_type: StaticString,
        learning_rate: Scalar[DType.float32],
        warmup_steps: Int,
        train_num_batches: Int,
        final_learning_rate_fraction: Scalar[DType.float32],
        cooldown_fraction: Scalar[DType.float32] = 0.2,
        decay_shape: StaticString = "sqrt",
    ):
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.train_num_batches = train_num_batches
        self.final_learning_rate_fraction = final_learning_rate_fraction
        self.cooldown_fraction = cooldown_fraction
        self.decay_shape = decay_shape

    def get_learning_rate_cosine(self, step: Int) -> Scalar[DType.float32]:
        """
        Cosine: Warmup linearly to the max learning rate, then decay cosine to the final learning rate (lr * final_learning_rate_fraction).
        """
        var lr: Scalar[DType.float32] = self.learning_rate
        if step < self.warmup_steps:
            lr = (
                lr
                * Scalar[DType.float32](step + 1)
                / Scalar[DType.float32](self.warmup_steps)
            )
            return lr
        var decay_ratio = Scalar[DType.float32](
            step - self.warmup_steps
        ) / Scalar[DType.float32](self.train_num_batches - self.warmup_steps)
        var coeff = Scalar[DType.float32](0.5) * (
            Scalar[DType.float32](1.0) + cos(pi * decay_ratio)
        )
        var min_lr: Scalar[DType.float32] = (
            self.learning_rate * self.final_learning_rate_fraction
        )
        lr = min_lr + coeff * (lr - min_lr) / Scalar[DType.float32](1.0)
        return lr

    def get_learning_rate_linear(self, step: Int) -> Scalar[DType.float32]:
        """
        Linear: Warmup linearly to the max learning rate, then decay linearly to the final learning rate (lr * final_learning_rate_fraction).
        """
        var lr: Scalar[DType.float32] = self.learning_rate
        if step < self.warmup_steps:
            lr = (
                lr
                * Scalar[DType.float32](step + 1)
                / Scalar[DType.float32](self.warmup_steps)
            )
            return lr
        var decay_ratio = Scalar[DType.float32](
            step - self.warmup_steps
        ) / Scalar[DType.float32](self.train_num_batches - self.warmup_steps)
        var min_lr: Scalar[DType.float32] = (
            self.learning_rate * self.final_learning_rate_fraction
        )
        lr = min_lr + decay_ratio * (lr - min_lr)
        return lr

    def get_learning_rate_constant(self, step: Int) -> Scalar[DType.float32]:
        """
        Constant: Return the max learning rate.
        """
        return self.learning_rate

    def get_learning_rate_wsd(self, step: Int) -> Scalar[DType.float32]:
        """
        WSD: warmup linearly, keep constant, the last `cooldown_fraction` of training decays (1 - sqrt or linear, per `decay_shape`) to the final fraction. Defaults (0.2, sqrt) are the original curve. Warmup 0 is valid.
        """
        # See https://arxiv.org/abs/2405.18392
        var lr: Scalar[DType.float32] = self.learning_rate
        var max_lr: Scalar[DType.float32] = lr
        # floor(N * (1 - cooldown)). float32 0.2 is 0.2 + 3e-9, so nudge the
        # product up by a relative 1e-6 to keep 0.2 -> N*4//5 exact.
        var n64 = Float64(self.train_num_batches)
        var decay_point = Int(
            n64 * (1.0 - Float64(self.cooldown_fraction)) + n64 * 1e-6 + 1e-9
        )

        if step < self.warmup_steps:
            var decay_ratio = Scalar[DType.float32](step + 1) / Scalar[
                DType.float32
            ](self.warmup_steps)
            lr = max_lr * decay_ratio
            return lr

        if step < decay_point:
            return lr

        var decay_ratio: Scalar[DType.float32] = Scalar[DType.float32](
            step - decay_point
        ) / Scalar[DType.float32](max(self.train_num_batches - decay_point, 1))
        var min_lr: Scalar[DType.float32] = (
            max_lr * self.final_learning_rate_fraction
        )
        if self.decay_shape == "linear":
            return min_lr + (1.0 - decay_ratio) * (max_lr - min_lr)
        return min_lr + (1.0 - sqrt(decay_ratio)) * (max_lr - min_lr)

    def get_learning_rate(self, step: Int) raises -> Scalar[DType.float32]:
        if self.scheduler_type == "cosine":
            return self.get_learning_rate_cosine(step)
        elif self.scheduler_type == "linear":
            return self.get_learning_rate_linear(step)
        elif self.scheduler_type == "constant":
            return self.get_learning_rate_constant(step)
        elif self.scheduler_type == "wsd":
            return self.get_learning_rate_wsd(step)
        else:
            raise Error("Invalid scheduler type: " + self.scheduler_type)
