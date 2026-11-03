# Graph Report - UAM-AD  (2026-11-02)

## Corpus Check
- 228 files · ~431,258 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 149 file(s) not represented in the graph (top: .log 66, .ckpt 64, .npz 18)

## Summary
- 1020 nodes · 2202 edges · 46 communities (36 shown, 10 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 103 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Phase 3 DTL Aggregation
- SN Session Building
- Data Loading & Features
- BaseModel Scoring
- Cost Measurement
- MultiDiscriminator
- DeepTraLog Evaluation
- Dataset & SOTA Choice
- Optimizer & Embeddings
- Seed Aggregation & ART Runner
- AnoFusion Model
- RE2-OB Preprocessing
- AnoFusion Runner
- Main Architecture Diagram
- Competitor Models
- Item 10 Rescoring
- Eadro Runner
- BaseV3 Utilities
- Attention Modules
- RCAEval Preprocessing Helpers
- Low Encoders
- Log Encoders
- GGNN Runner
- KPI/Log Models
- DTL Oracle Rescoring
- Async Trace Branch
- Decoder Heads
- RE2/RE3-OB Results
- RCAEval Preprocessing Docs
- Phase 3 Item 9 Runs
- Temporal Conv Branch
- MultiModel Generator
- Phase Diagram Losses
- SN Data Windows
- Positional Attention
- ConvNet
- Item 10 Aggregation
- RE3 Preprocessing
- Architecture Flow Doc
- Training/Inference Phases
- AnoMod-SN Faults
- CLAUDE.md Rules
- TADBench
- Trace Modeling

## God Nodes (most connected - your core abstractions)
1. `BaseModel` - 29 edges
2. `DeepTraLogPreprocessor` - 23 edges
3. `SNPreprocessor` - 23 edges
4. `main()` - 20 edges
5. `evaluate()` - 20 edges
6. `main()` - 18 edges
7. `SNData` - 15 edges
8. `MultiDiscriminator` - 15 edges
9. `RCAEvalOBPreprocessor` - 13 edges
10. `worker()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `AsyncTraceModel autoencoder (count, attribute, order decoders)` --implements--> `AsyncTraceModel`  [EXTRACTED]
  docs/model_architecture_flow_en.md → codes/models/async_trace_model_v3.py
- `RCA ground truth per dataset (parsed from sample id / scenario)` --references--> `scenario_from_test_pkl()`  [EXTRACTED]
  docs/root_cause_localization_en.md → codes/models/rca.py
- `Runtime opts: decomposed GAT attention and ZV dedupe` --references--> `TraceEncoder`  [EXTRACTED]
  docs/experiment_results_re2_ob_trace_vs_baseline_en.md → codes/models/trace_model_v3.py
- `Residual-gated trace fusion (README summary)` --semantically_similar_to--> `CHANGE 8: residual-gated trace fusion y = y_base + g*delta`  [INFERRED] [semantically similar]
  README.md → docs/model_architecture_flow_en.md
- `DeepTraLog result: full >= single branches, GGNN stronger on F02/F09` --semantically_similar_to--> `Result: both branches beat only-sync on F01/F02/F13 (3 seeds)`  [INFERRED] [semantically similar]
  working_history.md → docs/experiment_results_deeptralog_trace_both_sync_async_vs_only_sync_en.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Unsupervised L+M+T models used for fair conclusions** — chon_dataset_va_sota_uam_ad, chon_dataset_va_sota_art, chon_dataset_va_sota_anofusion, chon_dataset_va_sota_uac_ad [EXTRACTED 0.95]
- **Labeled reference models (not fair comparison)** — chon_dataset_va_sota_eadro, chon_dataset_va_sota_oredge, chon_dataset_va_sota_mstgad [EXTRACTED 0.90]
- **Final result tables A-E of thesis** — chon_dataset_va_sota_bang_a, chon_dataset_va_sota_bang_b, chon_dataset_va_sota_bang_c, chon_dataset_va_sota_bang_d, chon_dataset_va_sota_bang_e [EXTRACTED 0.95]
- **Residual-gated trace fusion: base decoder, delta head, trace gate** — docs_model_architecture_flow_en_change8_residual_gated, docs_model_architecture_flow_en_trace_gate, docs_model_architecture_flow_en_delta_head [EXTRACTED 0.95]
- **Trace discriminator heads (structural, attribute, async) in MultiDiscriminator** — docs_model_architecture_flow_en_change5_structural_disc, docs_model_architecture_flow_en_change7_attr_disc, docs_model_architecture_flow_en_change10_async_disc [EXTRACTED 0.95]
- **Competitor models compared against UAM-AD in item 9** — working_history_art, working_history_anofusion, working_history_eadro, working_history_oredge [EXTRACTED 0.95]
- **Log, metric and trace encoders feeding the fused decoder** — result_main_architecture_log_modeling, result_main_architecture_metric_modeling, result_main_architecture_node_embeddings, result_main_architecture_fused_decoder [EXTRACTED 1.00]
- **Adversarial generator-discriminator training loop** — result_main_architecture_generator_module, result_main_architecture_discriminator_module, result_main_architecture_loss_legend [EXTRACTED 1.00]
- **Training losses (reconstruction, contrastive, adversarial)** — result_main_phase_reconstruction_loss, result_main_phase_contrastive_loss, result_main_phase_adversarial_loss [EXTRACTED 0.95]
- **Inference flow** — result_main_phase_matched_pairs, result_main_phase_generator_module, result_main_phase_reconstructed_pairs, result_main_phase_anomaly_score [EXTRACTED 0.95]

## Communities (46 total, 10 thin omitted)

### Community 0 - "Phase 3 DTL Aggregation"
Cohesion: 0.05
Nodes (16): main(), read(), read_seeds(), build_baseline(), main(), root_url(), trace_signatures(), classify_edge() (+8 more)

### Community 1 - "SN Session Building"
Cohesion: 0.05
Nodes (14): build_sessions(), main(), verify(), _fault_window_for(), main(), _parse_log_ts(), SNPreprocessor, dump_params() (+6 more)

### Community 2 - "Data Loading & Features"
Cohesion: 0.05
Nodes (24): async_message_entries(), construct_unmatched_data(), get_features(), kpi_selection(), load_sessions(), myDataset, normalization(), normalize_data() (+16 more)

### Community 3 - "BaseModel Scoring"
Cohesion: 0.07
Nodes (22): BaseModel, pseudo_Dataset, compute_hit_rate_at_k(), rank_top_k_services(), main(), Context-normalised async count term, trace_err for F13 trips/left fail-fast sub-case, Where the gate acts: decoder, training loss, epoch selection, raw_sum (+14 more)

### Community 4 - "Cost Measurement"
Cohesion: 0.07
Nodes (24): common(), finish(), n_params(), Peak, rss_bytes(), sn_test_stem(), timed(), train_scale() (+16 more)

### Community 5 - "MultiDiscriminator"
Cohesion: 0.05
Nodes (40): MultiDiscriminator, Standard evaluation protocol doc, AUROC and AUPRC secondary metrics, Primary metric: F1/P/R at val threshold (no point_adjust), Oracle F1 (threshold swept on test, point_adjust), 3-5 seeds reporting rule, Split rules: train / val (normal only) / test per dataset, Threshold = p95 of val normal scores (+32 more)

### Community 6 - "DeepTraLog Evaluation"
Cohesion: 0.15
Nodes (16): _find_scenario_files(), main(), _scan_latest_results(), _find_scenario_files(), main(), _find_scenario_files(), main(), _find_scenario_files() (+8 more)

### Community 7 - "Dataset & SOTA Choice"
Cohesion: 0.06
Nodes (41): Chon dataset va SOTA cho UAM-AD (thesis experiment design and results), AnoFusion (KDD 2023, unsupervised L+M+T), AnoFusion cost scales with N^3 channels (slow inference, RE3-TT infeasible), AnoMod-TT dataset (dropped 10-13), ART (ASE 2024, unsupervised L+M+T), Async discriminator head (CHANGE 10), AUROC and AUPRC (threshold-free metrics), Bang A dataset statistics (+33 more)

### Community 8 - "Optimizer & Embeddings"
Cohesion: 0.07
Nodes (14): AdaBelief, Embed, FitsLegendre, LinearAttention, OrEdge, batch(), log_counts(), main() (+6 more)

### Community 9 - "Seed Aggregation & ART Runner"
Cohesion: 0.11
Nodes (18): mean_std(), print_table(), batch(), Data, load(), main(), prepare(), metric_index() (+10 more)

### Community 10 - "AnoFusion Model"
Cohesion: 0.09
Nodes (8): AnoFusionNet, GATGRU, GATNet, GraphAttentionLayer, GTAggregation, GTLayer, GTN, row_norm()

### Community 11 - "RE2-OB Preprocessing"
Cohesion: 0.12
Nodes (3): main(), RCAEvalOBPreprocessor, main()

### Community 12 - "AnoFusion Runner"
Cohesion: 0.15
Nodes (11): build_channels(), build_vocab(), load(), main(), make_batch(), mi_matrices(), predict_err(), channels() (+3 more)

### Community 13 - "Main Architecture Diagram"
Cohesion: 0.10
Nodes (25): Main Architecture Diagram (Generator-Discriminator multimodal AD), Attribute decoder, Discriminator heads (1D CNN + Classification layer, Real or reconstructed, x4), Discriminator module, Fused decoder, GAT encoder, Generator module, Log modeling (1D CNN + Trans_encoder) (+17 more)

### Community 14 - "Competitor Models"
Cohesion: 0.15
Nodes (3): async_encoder_inputs(), AsyncTraceModel, TraceEncoder

### Community 15 - "Item 10 Rescoring"
Cohesion: 0.13
Nodes (15): checkpoint_dir(), orchestrate(), out_path(), run_py_args(), worker(), collect(), rescore(), rank_hits() (+7 more)

### Community 16 - "Eadro Runner"
Cohesion: 0.14
Nodes (12): add_logs(), build(), batch(), main(), make(), predict(), run_number(), main() (+4 more)

### Community 17 - "BaseV3 Utilities"
Cohesion: 0.11
Nodes (8): find_positive_segment(), plot_labeled_2curve(), plot_labeled_curve(), Loss_fuse_model, KpiDiscriminator, KpiModel, LogDiscriminator, LogModel

### Community 18 - "Attention Modules"
Cohesion: 0.12
Nodes (6): AddAttention, DotAttention, MultiEncoder, KpiEncoder, MultiHeadAttention, ScaledDotProductAttention

### Community 20 - "Low Encoders"
Cohesion: 0.13
Nodes (4): MultiEncoder_low, KpiEncoder_low, GATLayer, TraceEncoder_low

### Community 21 - "Log Encoders"
Cohesion: 0.13
Nodes (5): LogEncoder, LogEncoder_low, PositionalEmbedding, Return0, Trans

### Community 22 - "GGNN Runner"
Cohesion: 0.18
Nodes (9): batches(), embed(), GGNN, Graphs, main(), metrics(), oracle_f1(), run_seed() (+1 more)

### Community 23 - "KPI/Log Models"
Cohesion: 0.21
Nodes (3): Decoder, Embedder, SelfAttention

### Community 24 - "DTL Oracle Rescoring"
Cohesion: 0.21
Nodes (9): checkpoint_dir(), orchestrate(), rebuild(), _score_components(), result_path(), run_py_args(), stored_scores(), worker() (+1 more)

### Community 25 - "Async Trace Branch"
Cohesion: 0.14
Nodes (15): DeepTraLog: sync-only vs sync+async results doc, Result: both branches beat only-sync on F01/F02/F13 (3 seeds), AsyncTraceModel autoencoder (count, attribute, order decoders), async_encoder_inputs (one-hot node identity, order relation excluded from mask), Async branch is only requested; Process.async_usable gate (>=40 steps), CHANGE 9: additive async trace branch, DeepTraLog preprocessing doc, Async fault vs structural (sync-type) fault definition (+7 more)

### Community 26 - "Decoder Heads"
Cohesion: 0.18
Nodes (4): Return0, Return1, ReturnSelf, ReturnTopXWeight

### Community 27 - "RE2/RE3-OB Results"
Cohesion: 0.15
Nodes (14): RE2-OB trace vs baseline results doc, Runtime opts: decomposed GAT attention and ZV dedupe, RE2-OB: trace wins 6/6 (mean F1 0.703 to 0.800), RE3-OB trace vs baseline results doc, RE3-OB: gated trace within 0.007 F1 of baseline (raw_sum), gate_delta_lr_mult (10x lr for zero-init gate and delta head), async_trace_gate (own gate, bias init 0.0), CHANGE 4: variance-based alpha (replaces learnable trace_alpha) (+6 more)

### Community 28 - "RCAEval Preprocessing Docs"
Cohesion: 0.17
Nodes (13): Fault category compatibility with trace branch, RE2-OB preprocessing doc, Canonical metric columns (union of 86, missing filled 0), RE2-OB dataset (infrastructure faults, 30 scenarios x 3 runs), Log token fallback to container_name when log_template missing, RE3-OB preprocessing doc, RE3-OB dataset (code-defect faults f1-f5), Drain3 two-pass log template extraction (+5 more)

### Community 29 - "Phase 3 Item 9 Runs"
Cohesion: 0.26
Nodes (6): jobs(), main(), tests_of(), features(), main(), score()

### Community 30 - "Temporal Conv Branch"
Cohesion: 0.17
Nodes (4): Branch, Chomp1d, ConvNet, SelfAttention

### Community 32 - "Phase Diagram Losses"
Cohesion: 0.32
Nodes (12): Adversarial loss, Anomaly score, Contrastive loss, Discriminator module, Generator module, Matched pairs, Real (discriminator output), Reconstructed (discriminator output) (+4 more)

### Community 34 - "Positional Attention"
Cohesion: 0.20
Nodes (3): AddAttention, PositionalEmbedding, Return0

### Community 35 - "ConvNet"
Cohesion: 0.22
Nodes (3): Chomp1d, ConvNet, TemporalEncoder

### Community 37 - "RE3 Preprocessing"
Cohesion: 0.25
Nodes (3): _build_drain3_miner(), main(), main()

### Community 38 - "Architecture Flow Doc"
Cohesion: 0.33
Nodes (6): Model architecture and data flow doc, CHANGE 1: trace separated from self-attention, CHANGE 2: ZV injected into decoder (TraceDAE Eq.10 inspired), Contrastive hinge loss on unmatched KPI pairs, GAN adversarial training (generator/discriminator steps), Generator MultiModel (log+KPI+trace autoencoder)

### Community 40 - "Training/Inference Phases"
Cohesion: 0.67
Nodes (3): Main Phase Diagram (training and inferring), Infering phase, Training phase

## Knowledge Gaps
- **59 isolated node(s):** `MSTGAD (semi-supervised, same family as OrEdge)`, `Hades (ICSE 2023, semi-supervised L+M)`, `TADBench (trace anomaly baselines collection)`, `AnoMod-SN dataset (SocialNetwork, 12 scenarios)`, `RCAEval RE2-OB (Online Boutique, infra faults)` (+54 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 362 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseModel` connect `BaseModel Scoring` to `BaseV3 Utilities`, `Data Loading & Features`, `MultiDiscriminator`, `MultiModel Generator`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **What connects `MSTGAD (semi-supervised, same family as OrEdge)`, `Hades (ICSE 2023, semi-supervised L+M)`, `TADBench (trace anomaly baselines collection)` to the rest of the system?**
  _59 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Phase 3 DTL Aggregation` be split into smaller, more focused modules?**
  _Cohesion score 0.05285592497868713 - nodes in this community are weakly interconnected._
- **Why does `MultiDiscriminator` connect `MultiDiscriminator` to `BaseV3 Utilities`, `BaseModel Scoring`, `Low Encoders`, `Competitor Models`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Should `SN Session Building` be split into smaller, more focused modules?**
  _Cohesion score 0.05128205128205128 - nodes in this community are weakly interconnected._
- **Should `Data Loading & Features` be split into smaller, more focused modules?**
  _Cohesion score 0.05288207297726071 - nodes in this community are weakly interconnected._
- **Should `BaseModel Scoring` be split into smaller, more focused modules?**
  _Cohesion score 0.06787330316742081 - nodes in this community are weakly interconnected._