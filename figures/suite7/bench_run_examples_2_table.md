| Mode | Throughput (tok/s) | Speedup vs full | KV cache (bytes) | KV max budget (bytes) | Token match vs full |
|---|---:|---:|---:|---:|---:|
| Full recompute | 2.061 | 1.000× | 0 | — | — |
| KV cache (FP) | 2.250 | 1.092× | 995,104 | 0 | 1.0000 |
| KV int8 | 2.243 | 1.088× | 251,712 | 0 | 1.0000 |
| KV int4 | 2.232 | 1.083× | 127,808 | 0 | 0.9458 |
| KV FP + budget | 2.390 | 1.159× | 123,360 | 131,072 | 0.6583 |
| KV int8 + budget | 2.355 | 1.143× | 131,072 | 131,072 | 0.8000 |
| KV int4 + budget | 2.371 | 1.150× | 127,808 | 131,072 | 0.9458 |
