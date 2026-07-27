//! Mutation rule types: concrete topology changes with audit trails.
//!
//! Five core rules for Phase 3:
//! 1. AddNode: introduce a new node with a label
//! 2. RemoveNode: delete a node (if no incoming edges)
//! 3. AddEdge: connect two nodes
//! 4. RemoveEdge: disconnect two nodes
//! 5. RewireEdge: change an edge's target

use super::super::error::NtgError;
use super::super::graph::{Graph, NodeId, NodeKind};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MutationRuleKind {
    // Topology mutations (Phase 6.0-6.5)
    AddNode { label: String },
    RemoveNode { node_id: NodeId },
    AddEdge { from: NodeId, to: NodeId },
    RemoveEdge { from: NodeId, to: NodeId },
    RewireEdge {
        from: NodeId,
        old_to: NodeId,
        new_to: NodeId,
    },

    // Web Design mutations (Phase 6.7)
    AdjustColorContrast { increase: bool },
    ReorderCTA { position: CTAPosition },
    SimplifyLayout { target_elements: usize },
    AdjustTypographyHierarchy { emphasis_level: u8 },
    ChangeFormFields { num_fields: usize },
    ModifyImagePlacement { placement: ImagePlacement },
    AdjustWhitespace { spacing_ratio: f32 },

    // Marketing mutations (Phase 6.7)
    RefocusValueProposition { focus_area: String },
    AdjustCopyTone { tone: CopyTone },
    ChangeTargetAudience { segment: AudienceSegment },
    ModifyPricingTier { tier_index: usize },
    ShiftChannelMix { primary_channel: String },
    AdjustRetentionStrategy { strategy: RetentionStrategy },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CTAPosition {
    AboveFold,
    BelowFold,
    FloatingCorner,
    Inline,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImagePlacement {
    TopCenter,
    LeftAligned,
    RightAligned,
    FullWidth,
    Hidden,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CopyTone {
    Authoritative,
    Relateable,
    Urgent,
    Educational,
    Playful,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AudienceSegment {
    Enterprise,
    Startup,
    SMB,
    Individual,
    Niche(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RetentionStrategy {
    Gamification,
    EmailNurture,
    CommunityBuilding,
    PremiumContent,
    PersonalizedRecommendations,
}

impl MutationRuleKind {
    /// Get a human-readable description of this mutation.
    pub fn description(&self) -> String {
        match self {
            // Topology mutations
            MutationRuleKind::AddNode { label } => format!("add_node(label='{}')", label),
            MutationRuleKind::RemoveNode { node_id } => format!("remove_node(id={})", node_id),
            MutationRuleKind::AddEdge { from, to } => format!("add_edge({}→{})", from, to),
            MutationRuleKind::RemoveEdge { from, to } => format!("remove_edge({}→{})", from, to),
            MutationRuleKind::RewireEdge {
                from,
                old_to,
                new_to,
            } => format!("rewire_edge({}→{} → {}→{})", from, old_to, from, new_to),

            // Web Design mutations
            MutationRuleKind::AdjustColorContrast { increase } => {
                format!("adjust_color_contrast(increase={})", increase)
            }
            MutationRuleKind::ReorderCTA { position } => {
                format!("reorder_cta(position={:?})", position)
            }
            MutationRuleKind::SimplifyLayout { target_elements } => {
                format!("simplify_layout(target_elements={})", target_elements)
            }
            MutationRuleKind::AdjustTypographyHierarchy { emphasis_level } => {
                format!("adjust_typography(emphasis={})", emphasis_level)
            }
            MutationRuleKind::ChangeFormFields { num_fields } => {
                format!("change_form_fields(num_fields={})", num_fields)
            }
            MutationRuleKind::ModifyImagePlacement { placement } => {
                format!("modify_image_placement(placement={:?})", placement)
            }
            MutationRuleKind::AdjustWhitespace { spacing_ratio } => {
                format!("adjust_whitespace(ratio={:.2})", spacing_ratio)
            }

            // Marketing mutations
            MutationRuleKind::RefocusValueProposition { focus_area } => {
                format!("refocus_value_prop(focus='{}')", focus_area)
            }
            MutationRuleKind::AdjustCopyTone { tone } => {
                format!("adjust_copy_tone(tone={:?})", tone)
            }
            MutationRuleKind::ChangeTargetAudience { segment } => {
                format!("change_audience(segment={:?})", segment)
            }
            MutationRuleKind::ModifyPricingTier { tier_index } => {
                format!("modify_pricing(tier={})", tier_index)
            }
            MutationRuleKind::ShiftChannelMix { primary_channel } => {
                format!("shift_channel(primary='{}')", primary_channel)
            }
            MutationRuleKind::AdjustRetentionStrategy { strategy } => {
                format!("adjust_retention(strategy={:?})", strategy)
            }
        }
    }
}

/// A versioned, auditable mutation rule.
#[derive(Clone, Debug)]
pub struct MutationRule {
    pub kind: MutationRuleKind,
}

impl MutationRule {
    pub fn description(&self) -> String {
        match &self.kind {
            // Topology mutations
            MutationRuleKind::AddNode { label } => format!("add_node(label='{}')", label),
            MutationRuleKind::RemoveNode { node_id } => format!("remove_node(id={})", node_id),
            MutationRuleKind::AddEdge { from, to } => format!("add_edge({}→{})", from, to),
            MutationRuleKind::RemoveEdge { from, to } => format!("remove_edge({}→{})", from, to),
            MutationRuleKind::RewireEdge {
                from,
                old_to,
                new_to,
            } => format!("rewire_edge({}→{} → {}→{})", from, old_to, from, new_to),

            // Web Design mutations
            MutationRuleKind::AdjustColorContrast { increase } => {
                format!("adjust_color_contrast(increase={})", increase)
            }
            MutationRuleKind::ReorderCTA { position } => {
                format!("reorder_cta(position={:?})", position)
            }
            MutationRuleKind::SimplifyLayout { target_elements } => {
                format!("simplify_layout(target_elements={})", target_elements)
            }
            MutationRuleKind::AdjustTypographyHierarchy { emphasis_level } => {
                format!("adjust_typography(emphasis={})", emphasis_level)
            }
            MutationRuleKind::ChangeFormFields { num_fields } => {
                format!("change_form_fields(num_fields={})", num_fields)
            }
            MutationRuleKind::ModifyImagePlacement { placement } => {
                format!("modify_image_placement(placement={:?})", placement)
            }
            MutationRuleKind::AdjustWhitespace { spacing_ratio } => {
                format!("adjust_whitespace(ratio={:.2})", spacing_ratio)
            }

            // Marketing mutations
            MutationRuleKind::RefocusValueProposition { focus_area } => {
                format!("refocus_value_prop(focus='{}')", focus_area)
            }
            MutationRuleKind::AdjustCopyTone { tone } => {
                format!("adjust_copy_tone(tone={:?})", tone)
            }
            MutationRuleKind::ChangeTargetAudience { segment } => {
                format!("change_audience(segment={:?})", segment)
            }
            MutationRuleKind::ModifyPricingTier { tier_index } => {
                format!("modify_pricing(tier={})", tier_index)
            }
            MutationRuleKind::ShiftChannelMix { primary_channel } => {
                format!("shift_channel(primary='{}')", primary_channel)
            }
            MutationRuleKind::AdjustRetentionStrategy { strategy } => {
                format!("adjust_retention(strategy={:?})", strategy)
            }
        }
    }

    /// Apply this rule to a graph (creates a test topology).
    /// The graph must be cloned before calling this so the original is unchanged.
    ///
    /// Note: Web design and marketing mutations are no-ops here; they are evaluated
    /// by domain-specific evaluators that measure real metrics (conversion, CTR, etc.)
    pub fn apply(&self, graph: &mut Graph) -> Result<(), NtgError> {
        match &self.kind {
            // Topology mutations
            MutationRuleKind::AddNode { label } => {
                // Content node by default; Execution nodes are a separate concern.
                let _id = graph.add_node(NodeKind::Content, label.clone());
                Ok(())
            }
            MutationRuleKind::RemoveNode { node_id } => {
                graph.remove_node(*node_id)?;
                Ok(())
            }
            MutationRuleKind::AddEdge { from, to } => {
                graph.add_edge(*from, *to)?;
                Ok(())
            }
            MutationRuleKind::RemoveEdge { from, to } => {
                graph.remove_edge(*from, *to)?;
                Ok(())
            }
            MutationRuleKind::RewireEdge {
                from,
                old_to,
                new_to,
            } => {
                graph.remove_edge(*from, *old_to)?;
                graph.add_edge(*from, *new_to)?;
                Ok(())
            }

            // Web Design mutations (no-op on graph; evaluated via domain metrics)
            MutationRuleKind::AdjustColorContrast { .. } => Ok(()),
            MutationRuleKind::ReorderCTA { .. } => Ok(()),
            MutationRuleKind::SimplifyLayout { .. } => Ok(()),
            MutationRuleKind::AdjustTypographyHierarchy { .. } => Ok(()),
            MutationRuleKind::ChangeFormFields { .. } => Ok(()),
            MutationRuleKind::ModifyImagePlacement { .. } => Ok(()),
            MutationRuleKind::AdjustWhitespace { .. } => Ok(()),

            // Marketing mutations (no-op on graph; evaluated via domain metrics)
            MutationRuleKind::RefocusValueProposition { .. } => Ok(()),
            MutationRuleKind::AdjustCopyTone { .. } => Ok(()),
            MutationRuleKind::ChangeTargetAudience { .. } => Ok(()),
            MutationRuleKind::ModifyPricingTier { .. } => Ok(()),
            MutationRuleKind::ShiftChannelMix { .. } => Ok(()),
            MutationRuleKind::AdjustRetentionStrategy { .. } => Ok(()),
        }
    }

    /// Versioned rule ID (for audit trails).
    /// In a real system, this would be a content hash or semantic versioning.
    pub fn rule_version(&self) -> u32 {
        1 // Phase 3.0: all rules versioned as 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_description_add_node() {
        let rule = MutationRule {
            kind: MutationRuleKind::AddNode {
                label: "test_node".to_string(),
            },
        };
        assert_eq!(rule.description(), "add_node(label='test_node')");
    }

    #[test]
    fn rule_description_remove_node() {
        let rule = MutationRule {
            kind: MutationRuleKind::RemoveNode { node_id: 42 },
        };
        assert_eq!(rule.description(), "remove_node(id=42)");
    }

    #[test]
    fn rule_description_add_edge() {
        let rule = MutationRule {
            kind: MutationRuleKind::AddEdge { from: 1, to: 2 },
        };
        assert_eq!(rule.description(), "add_edge(1→2)");
    }

    #[test]
    fn rule_description_remove_edge() {
        let rule = MutationRule {
            kind: MutationRuleKind::RemoveEdge { from: 1, to: 2 },
        };
        assert_eq!(rule.description(), "remove_edge(1→2)");
    }

    #[test]
    fn rule_description_rewire_edge() {
        let rule = MutationRule {
            kind: MutationRuleKind::RewireEdge {
                from: 1,
                old_to: 2,
                new_to: 3,
            },
        };
        assert_eq!(rule.description(), "rewire_edge(1→2 → 1→3)");
    }

    #[test]
    fn rule_version_is_one() {
        let rule = MutationRule {
            kind: MutationRuleKind::AddNode {
                label: "test".to_string(),
            },
        };
        assert_eq!(rule.rule_version(), 1);
    }

    #[test]
    fn apply_add_node() -> Result<(), NtgError> {
        let mut graph = Graph::new();
        let rule = MutationRule {
            kind: MutationRuleKind::AddNode {
                label: "new_node".to_string(),
            },
        };
        rule.apply(&mut graph)?;
        // Verify node was added (basic smoke test)
        assert!(graph.node_count() > 0);
        Ok(())
    }

    // Web Design mutation tests
    #[test]
    fn rule_description_adjust_color_contrast() {
        let rule = MutationRule {
            kind: MutationRuleKind::AdjustColorContrast { increase: true },
        };
        assert_eq!(rule.description(), "adjust_color_contrast(increase=true)");
    }

    #[test]
    fn rule_description_reorder_cta() {
        let rule = MutationRule {
            kind: MutationRuleKind::ReorderCTA {
                position: CTAPosition::FloatingCorner,
            },
        };
        assert!(rule.description().contains("reorder_cta"));
        assert!(rule.description().contains("FloatingCorner"));
    }

    #[test]
    fn rule_description_simplify_layout() {
        let rule = MutationRule {
            kind: MutationRuleKind::SimplifyLayout {
                target_elements: 5,
            },
        };
        assert_eq!(rule.description(), "simplify_layout(target_elements=5)");
    }

    #[test]
    fn rule_description_adjust_typography() {
        let rule = MutationRule {
            kind: MutationRuleKind::AdjustTypographyHierarchy {
                emphasis_level: 3,
            },
        };
        assert_eq!(rule.description(), "adjust_typography(emphasis=3)");
    }

    #[test]
    fn rule_description_change_form_fields() {
        let rule = MutationRule {
            kind: MutationRuleKind::ChangeFormFields { num_fields: 3 },
        };
        assert_eq!(rule.description(), "change_form_fields(num_fields=3)");
    }

    #[test]
    fn rule_description_modify_image_placement() {
        let rule = MutationRule {
            kind: MutationRuleKind::ModifyImagePlacement {
                placement: ImagePlacement::FullWidth,
            },
        };
        assert!(rule.description().contains("modify_image_placement"));
        assert!(rule.description().contains("FullWidth"));
    }

    #[test]
    fn rule_description_adjust_whitespace() {
        let rule = MutationRule {
            kind: MutationRuleKind::AdjustWhitespace {
                spacing_ratio: 1.25,
            },
        };
        assert_eq!(rule.description(), "adjust_whitespace(ratio=1.25)");
    }

    // Marketing mutation tests
    #[test]
    fn rule_description_refocus_value_prop() {
        let rule = MutationRule {
            kind: MutationRuleKind::RefocusValueProposition {
                focus_area: "cost_savings".to_string(),
            },
        };
        assert_eq!(
            rule.description(),
            "refocus_value_prop(focus='cost_savings')"
        );
    }

    #[test]
    fn rule_description_adjust_copy_tone() {
        let rule = MutationRule {
            kind: MutationRuleKind::AdjustCopyTone {
                tone: CopyTone::Urgent,
            },
        };
        assert!(rule.description().contains("adjust_copy_tone"));
        assert!(rule.description().contains("Urgent"));
    }

    #[test]
    fn rule_description_change_target_audience() {
        let rule = MutationRule {
            kind: MutationRuleKind::ChangeTargetAudience {
                segment: AudienceSegment::Enterprise,
            },
        };
        assert!(rule.description().contains("change_audience"));
        assert!(rule.description().contains("Enterprise"));
    }

    #[test]
    fn rule_description_modify_pricing_tier() {
        let rule = MutationRule {
            kind: MutationRuleKind::ModifyPricingTier { tier_index: 2 },
        };
        assert_eq!(rule.description(), "modify_pricing(tier=2)");
    }

    #[test]
    fn rule_description_shift_channel_mix() {
        let rule = MutationRule {
            kind: MutationRuleKind::ShiftChannelMix {
                primary_channel: "email".to_string(),
            },
        };
        assert_eq!(rule.description(), "shift_channel(primary='email')");
    }

    #[test]
    fn rule_description_adjust_retention_strategy() {
        let rule = MutationRule {
            kind: MutationRuleKind::AdjustRetentionStrategy {
                strategy: RetentionStrategy::Gamification,
            },
        };
        assert!(rule.description().contains("adjust_retention"));
        assert!(rule.description().contains("Gamification"));
    }

    #[test]
    fn apply_web_design_mutations() -> Result<(), NtgError> {
        let mut graph = Graph::new();

        // All web design mutations should be no-ops on graph
        let mutations = vec![
            MutationRule {
                kind: MutationRuleKind::AdjustColorContrast { increase: true },
            },
            MutationRule {
                kind: MutationRuleKind::ReorderCTA {
                    position: CTAPosition::AboveFold,
                },
            },
            MutationRule {
                kind: MutationRuleKind::SimplifyLayout {
                    target_elements: 5,
                },
            },
            MutationRule {
                kind: MutationRuleKind::AdjustTypographyHierarchy {
                    emphasis_level: 2,
                },
            },
            MutationRule {
                kind: MutationRuleKind::ChangeFormFields { num_fields: 3 },
            },
            MutationRule {
                kind: MutationRuleKind::ModifyImagePlacement {
                    placement: ImagePlacement::RightAligned,
                },
            },
            MutationRule {
                kind: MutationRuleKind::AdjustWhitespace {
                    spacing_ratio: 1.5,
                },
            },
        ];

        for mutation in mutations {
            mutation.apply(&mut graph)?;
        }

        Ok(())
    }

    #[test]
    fn apply_marketing_mutations() -> Result<(), NtgError> {
        let mut graph = Graph::new();

        // All marketing mutations should be no-ops on graph
        let mutations = vec![
            MutationRule {
                kind: MutationRuleKind::RefocusValueProposition {
                    focus_area: "speed".to_string(),
                },
            },
            MutationRule {
                kind: MutationRuleKind::AdjustCopyTone {
                    tone: CopyTone::Authoritative,
                },
            },
            MutationRule {
                kind: MutationRuleKind::ChangeTargetAudience {
                    segment: AudienceSegment::Startup,
                },
            },
            MutationRule {
                kind: MutationRuleKind::ModifyPricingTier { tier_index: 1 },
            },
            MutationRule {
                kind: MutationRuleKind::ShiftChannelMix {
                    primary_channel: "paid_ads".to_string(),
                },
            },
            MutationRule {
                kind: MutationRuleKind::AdjustRetentionStrategy {
                    strategy: RetentionStrategy::EmailNurture,
                },
            },
        ];

        for mutation in mutations {
            mutation.apply(&mut graph)?;
        }

        Ok(())
    }
}
