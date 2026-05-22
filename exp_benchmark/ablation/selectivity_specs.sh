#!/usr/bin/env bash
# selectivity_specs.sh — shared DNF predicate specs for the 6 selectivity points
# used in the main results table and ablations.
#
# Each entry is "<target_pct>|<dnf_name>|<dnf_spec_json>".
# Predicates target attr_type=[0,1]: AND of a range bound on attr_0 and a
# label list on attr_1. The 'OR of ANDs' form lets us hit arbitrary
# combinations of selectivities.

# 6-point log-spread sweep (1%, 2%, 3%, 5%, 7%, 10%)
EMA_SEL6=(
    '1|or_T1|[{"0":0.1,"1":[9]},{"1":[12]}]'
    '2|or_T2|[{"0":0.1,"1":[8]},{"1":[12]}]'
    '3|or_T3|[{"0":0.15,"1":[8]},{"1":[12]}]'
    '5|or_T5|[{"0":0.177,"1":[7]},{"1":[12]}]'
    '7|or_T7|[{"0":0.233,"1":[7]},{"1":[12]}]'
    '10|or_T10|[{"0":0.3,"1":[9]},{"1":[12]}]'
)

# Extended high-selectivity sweep (20-100%) for M-axis ablation
EMA_SEL_HIGH=(
    '20|or_T20|[{"1":[8]}]'
    '40|or_T40|[{"1":[6]}]'
    '60|or_T60|[{"1":[4]}]'
    '80|or_T80|[{"1":[2]}]'
    '100|or_T100|[{"0":1.0}]'
)
