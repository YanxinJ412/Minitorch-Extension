| Mode | Throughput (tok/s) | Speedup vs full | KV cache (bytes) | KV max budget (bytes) | Token match vs full |
|---|---:|---:|---:|---:|---:|
| Full recompute | 2.214 | 1.000× | 0 | — | — |
| KV cache (FP) | 2.434 | 1.099× | 991,232 | — | 1.0000 |
| KV int8 | 2.430 | 1.098× | 247,840 | — | 1.0000 |
| KV int4 | 2.421 | 1.093× | 123,936 | — | 0.9458 |
