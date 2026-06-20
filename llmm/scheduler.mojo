from std.math import cos, pi, sqrt


struct LearningRateScheduler:
    var scheduler_type: StaticString
    var learning_rate: Scalar[DType.float32]
    var warmup_steps: Int
    var train_num_batches: Int
    var final_learning_rate_fraction: Scalar[DType.float32]

    def __init__(
        out self,
        scheduler_type: StaticString,
        learning_rate: Scalar[DType.float32],
        warmup_steps: Int,
        train_num_batches: Int,
        final_learning_rate_fraction: Scalar[DType.float32],
    ):
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.train_num_batches = train_num_batches
        self.final_learning_rate_fraction = final_learning_rate_fraction

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
        WSD: warmup linearly, keep constant, last 20 percent of training decay using 1 - sqrt decay to the final fraction (should be 0.0).
        """
        # See https://arxiv.org/abs/2405.18392
        var lr: Scalar[DType.float32] = self.learning_rate
        var max_lr: Scalar[DType.float32] = lr
        var decay_point = self.train_num_batches * 4 // 5

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
        ) / Scalar[DType.float32](self.train_num_batches - decay_point)
        var min_lr: Scalar[DType.float32] = (
            max_lr * self.final_learning_rate_fraction
        )
        lr = min_lr + (1.0 - sqrt(decay_ratio)) * (max_lr - min_lr)
        return lr

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
