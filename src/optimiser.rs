use rand_distr::num_traits::Pow;

use crate::layer::LayerData;

pub trait Optimiser {
    type Data<C: LayerData>: OptimiserData;
    fn update<C: LayerData>(&self, gradient: C, cache: &mut Self::Data<C>) -> C;
    fn finish_iteration(&mut self);
}

pub trait OptimiserData {
    fn empty() -> Self;
}

impl OptimiserData for () {
    fn empty() -> Self {}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicOptimiser {
    l_rate: f64,
    decay: f64,
}

impl BasicOptimiser {
    pub fn new(l_rate: f64, decay: f64) -> Self {
        Self { l_rate, decay }
    }
}

impl Optimiser for BasicOptimiser {
    type Data<C: LayerData> = ();

    fn update<C: LayerData>(&self, mut gradient: C, _cache: &mut ()) -> C {
        gradient.update_each(|f| f * self.l_rate);
        gradient
    }

    fn finish_iteration(&mut self) {
        self.l_rate *= 1. - self.decay;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumOptimiser {
    l_rate: f64,
    decay: f64,
    m_rate: f64,
}

impl MomentumOptimiser {
    pub fn new(l_rate: f64, decay: f64, m_rate: f64) -> Self {
        Self {
            l_rate,
            decay,
            m_rate,
        }
    }
}

impl Optimiser for MomentumOptimiser {
    type Data<C: LayerData> = MomentumCache<C>;

    fn update<C: LayerData>(&self, mut gradient: C, cache: &mut Self::Data<C>) -> C {
        cache.momentum.merge_each(&gradient, |mom, grad| {
            self.m_rate * mom + (1. - self.m_rate) * grad
        });
        gradient.merge_each(&cache.momentum, |_grad, mom| self.l_rate * mom);
        gradient
    }

    fn finish_iteration(&mut self) {
        self.l_rate *= 1. - self.decay;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MomentumCache<C: LayerData> {
    momentum: C,
}

impl<C: LayerData> OptimiserData for MomentumCache<C> {
    fn empty() -> Self {
        Self::new()
    }
}

impl<C: LayerData> MomentumCache<C> {
    pub fn new() -> Self {
        Self {
            momentum: C::zeroes(),
        }
    }
}

impl<C: LayerData> Default for MomentumCache<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RMSpropOptimiser {
    l_rate: f64,
    decay: f64,
    m_rate: f64,
}

impl RMSpropOptimiser {
    pub fn new(l_rate: f64, decay: f64, m_rate: f64) -> Self {
        Self {
            l_rate,
            decay,
            m_rate,
        }
    }
}

impl Optimiser for RMSpropOptimiser {
    type Data<C: LayerData> = MomentumCache<C>;

    fn update<C: LayerData>(&self, mut gradient: C, cache: &mut Self::Data<C>) -> C {
        cache.momentum.merge_each(&gradient, |mom, grad| {
            self.m_rate * mom + (1. - self.m_rate) * grad * grad
        });
        gradient.merge_each(&cache.momentum, |grad, mom| {
            self.l_rate * grad / (mom + f64::EPSILON).sqrt()
        });
        gradient
    }

    fn finish_iteration(&mut self) {
        self.l_rate *= 1. - self.decay;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamOptimiser {
    l_rate: f64,     // 0.001
    mean_decay: f64, // 0.9
    var_decay: f64,  // 0.999
    iteration: u32,
}

impl AdamOptimiser {
    pub fn new(l_rate: f64, mean_decay: f64, var_decay: f64) -> Self {
        Self {
            l_rate,
            mean_decay,
            var_decay,
            iteration: 1,
        }
    }

    pub fn new_default() -> Self {
        Self {
            l_rate: 0.001,
            mean_decay: 0.9,
            var_decay: 0.999,
            iteration: 1,
        }
    }
}

impl Optimiser for AdamOptimiser {
    type Data<C: LayerData> = AdamCache<C>;

    fn update<C: LayerData>(&self, mut gradient: C, cache: &mut Self::Data<C>) -> C {
        cache.mean.momentum.merge_each(&gradient, |mom, grad| {
            self.mean_decay * mom + (1. - self.mean_decay) * grad
        });
        cache.variance.momentum.merge_each(&gradient, |mom, grad| {
            self.var_decay * mom + (1. - self.var_decay) * grad * grad
        });

        gradient.merge_each(&cache.mean.momentum, |_grad, mom| {
            self.l_rate * (mom / (1. - self.mean_decay.pow(self.iteration as i32)))
        });
        gradient.merge_each(&cache.variance.momentum, |grad, mom| {
            grad / ((mom / (1. - self.var_decay.pow(self.iteration as i32))).sqrt() + f64::EPSILON)
        });
        gradient
    }

    fn finish_iteration(&mut self) {
        self.iteration += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdamCache<C: LayerData> {
    mean: MomentumCache<C>,
    variance: MomentumCache<C>,
}

impl<C: LayerData> OptimiserData for AdamCache<C> {
    fn empty() -> Self {
        Self::new()
    }
}

impl<C: LayerData> AdamCache<C> {
    pub fn new() -> Self {
        Self {
            mean: MomentumCache::new(),
            variance: MomentumCache::new(),
        }
    }
}

impl<C: LayerData> Default for AdamCache<C> {
    fn default() -> Self {
        Self::new()
    }
}
