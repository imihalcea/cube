mod column;
mod pass_through;
mod top_level;

use crate::{
    compile::rewrite::rewriter::{CubeRewrite, RewriteRules},
    config::ConfigObj,
};
use std::sync::Arc;

pub struct FlattenRules {
    config_obj: Arc<dyn ConfigObj>,
}

impl RewriteRules for FlattenRules {
    fn rewrite_rules(&self) -> Vec<CubeRewrite> {
        let mut rules = vec![];

        self.top_level_rules(&mut rules);
        self.pass_through_rules(&mut rules);
        self.column_rules(&mut rules);

        rules
    }
}

impl FlattenRules {
    pub fn new(config_obj: Arc<dyn ConfigObj>) -> Self {
        Self { config_obj }
    }
}
