#!/bin/bash
pod="interactive-torch-geometric"
dirname="CGGD"
kubectl exec --context 'bizon@sterling' -n bizon $pod -- mkdir -p /workspace/data/robokop/$dirname/raw
kubectl cp --context 'bizon@sterling' -n bizon robokop/$dirname/robo_train.txt $pod:/workspace/data/robokop/$dirname/raw/robo_train.txt
kubectl cp --context 'bizon@sterling' -n bizon robokop/$dirname/robo_test.txt $pod:/workspace/data/robokop/$dirname/raw/robo_test.txt
kubectl cp --context 'bizon@sterling' -n bizon robokop/$dirname/robo_val.txt $pod:/workspace/data/robokop/$dirname/raw/robo_valid.txt
