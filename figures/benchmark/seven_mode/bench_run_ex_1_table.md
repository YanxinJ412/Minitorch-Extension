| Mode | Throughput (tok/s) | Speedup vs full | KV cache (bytes) | KV max budget (bytes) | Token match vs full |
|---|---:|---:|---:|---:|---:|
| Full recompute | 2.061 | 1.000× | 0 | — | — |
| KV cache (FP) | 2.268 | 1.100× | 995,104 | 0 | 1.0000 |
| KV int8 | 2.260 | 1.097× | 251,712 | 0 | 1.0000 |
| KV int4 | 2.239 | 1.087× | 127,808 | 0 | 1.0000 |
| KV FP + budget | 2.276 | 1.105× | 123,360 | 131,072 | 1.0000 |
| KV int8 + budget | 2.250 | 1.092× | 131,072 | 131,072 | 1.0000 |
| KV int4 + budget | 2.247 | 1.090× | 127,808 | 131,072 | 1.0000 |
