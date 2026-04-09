# Environment Summary

This appendix-ready export captures the runtime context and the file inventory used to reproduce the benchmark outputs without relying on notebook state.

## Runtime

| key | value |
| --- | --- |
| generated_utc | 2026-04-09T12:07:18.954738+00:00 |
| python_version | 3.11.9 |
| python_executable | C:\Users\Safal Kumar\Desktop\synthetic-data-lifecycle-benchmarks\.venv\Scripts\python.exe |
| platform | Windows-10-10.0.26200-SP0 |
| system | Windows |
| release | 10 |
| machine | AMD64 |
| processor | AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD |
| cwd | C:\Users\Safal Kumar\Desktop\synthetic-data-lifecycle-benchmarks |
| timezone | N/A |
| cpu_count | 12 |
| pandas_version | 2.3.3 |
| numpy_version | 1.26.4 |
| matplotlib_version | 3.10.8 |
| git_branch | codex/reviewer-closure-implementation |
| git_commit | a445cc6af8a37d84aac26e3e10230768cdc0c043 |
| torch_version | 2.11.0+cu128 |
| cuda_available | Yes |
| cuda_device_count | 1 |

## Dataset Manifest

| dataset_name | domain | target_column | sensitive_attr | train_rows | test_rows | train_present | test_present |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | Socioeconomic | income | sex | 36177 | 9045 | Yes | Yes |
| bank | Marketing / Finance | target | age_group | 36168 | 9043 | Yes | Yes |
| covertype | Environmental / Ecological | Cover_Type | N/A | 46480 | 11621 | Yes | Yes |
| diabetes | Healthcare | target | age_group | 614 | 154 | Yes | Yes |

## Model Manifest

| model_id | display_name | available | trainable | adult_source_present | adult_source_bytes |
| --- | --- | --- | --- | --- | --- |
| ctgan | CTGAN | Yes | Yes | No | N/A |
| tvae | TVAE | Yes | Yes | No | N/A |
| tabddpm | TABDDPM | Yes | Yes | No | N/A |

## Artifact Inventory

| category | name | present | bytes | mtime_utc |
| --- | --- | --- | --- | --- |
| dataset | adult train | Yes | 3856157.0 | 2026-04-09T11:00:35.363004+00:00 |
| dataset | adult test | Yes | 964089.0 | 2026-04-09T11:00:35.383613+00:00 |
| synthetic | adult CTGAN | Yes | 1070987.0 | 2026-04-09T12:00:06.532756+00:00 |
| synthetic | adult TVAE | Yes | 1061182.0 | 2026-04-09T12:00:06.580245+00:00 |
| synthetic | adult TABDDPM | Yes | 1068427.0 | 2026-04-09T11:18:15.202126+00:00 |
| dataset | bank train | Yes | 2932371.0 | 2026-04-09T11:01:36.151532+00:00 |
| dataset | bank test | Yes | 733365.0 | 2026-04-09T11:01:36.173109+00:00 |
| synthetic | bank CTGAN | Yes | 809536.0 | 2026-04-09T12:00:06.557060+00:00 |
| synthetic | bank TVAE | Yes | 797929.0 | 2026-04-09T12:00:06.596081+00:00 |
| synthetic | bank TABDDPM | Yes | 811292.0 | 2026-04-09T11:23:40.491122+00:00 |
| dataset | covertype train | Yes | 6061027.0 | 2026-04-09T11:13:33.848312+00:00 |
| dataset | covertype test | Yes | 1515567.0 | 2026-04-09T11:13:33.899620+00:00 |
| synthetic | covertype CTGAN | Yes | 1301229.0 | 2026-04-09T12:05:59.603674+00:00 |
| synthetic | covertype TVAE | Yes | 1304425.0 | 2026-04-09T12:05:59.675060+00:00 |
| synthetic | covertype TABDDPM | Yes | 1303408.0 | 2026-04-09T11:38:32.984350+00:00 |
| dataset | diabetes train | Yes | 26816.0 | 2026-04-09T11:13:34.434813+00:00 |
| dataset | diabetes test | Yes | 6820.0 | 2026-04-09T11:13:34.434813+00:00 |
| synthetic | diabetes CTGAN | Yes | 1210752.0 | 2026-04-09T12:00:06.580245+00:00 |
| synthetic | diabetes TVAE | Yes | 425317.0 | 2026-04-09T12:00:06.612056+00:00 |
| synthetic | diabetes TABDDPM | Yes | 534216.0 | 2026-04-09T11:41:41.573057+00:00 |
| benchmark_results | cross-domain summary | Yes | 1168.0 | 2026-04-09T07:10:52.561543+00:00 |
| benchmark_results | mean rank table | Yes | 197.0 | 2026-04-09T07:10:52.561543+00:00 |
| benchmark_results | benchmark run notes | Yes | 2778.0 | 2026-04-09T07:10:52.561543+00:00 |
| benchmark_results | benchmark research summary | Yes | 4156.0 | 2026-04-09T07:10:52.561543+00:00 |
| benchmark_results | benchmark failures log | Yes | 903.0 | 2026-04-09T11:38:33.027433+00:00 |
| benchmark_results | cross-domain report | No | N/A | N/A |
| benchmark_results | compute summary | Yes | 3676.0 | 2026-04-09T12:07:18.545467+00:00 |
| benchmark_results | compute markdown | Yes | 1476.0 | 2026-04-09T12:07:18.545467+00:00 |
| benchmark_results | reproducibility manifest | No | N/A | N/A |
| benchmark_results | artifact inventory | No | N/A | N/A |
| dp_triangle | adult dp triangle dashboard | No | N/A | N/A |
| dp_triangle | adult direction3 findings | No | N/A | N/A |
| dp_triangle | adult direction3 methodology | No | N/A | N/A |

## Scope Notes

- Benchmark root: C:\Users\Safal Kumar\Desktop\synthetic-data-lifecycle-benchmarks\benchmarks
- Project root: C:\Users\Safal Kumar\Desktop\synthetic-data-lifecycle-benchmarks
- Missing optional files are recorded as absent rather than causing export failure.
