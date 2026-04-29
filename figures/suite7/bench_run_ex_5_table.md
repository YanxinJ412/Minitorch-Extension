| Mode | Throughput (tok/s) | Speedup vs full | KV cache (bytes) | KV max budget (bytes) | Token match vs full |
|---|---:|---:|---:|---:|---:|
| Full recompute | 2.160 | 1.000× | 0 | — | — |
| KV cache (FP) | 2.380 | 1.102× | 995,104 | 0 | 1.0000 |
| KV int8 | 2.368 | 1.096× | 251,712 | 0 | 0.5800 |
| KV int4 | 2.359 | 1.092× | 127,808 | 0 | 0.4650 |
| KV FP + budget | 2.382 | 1.103× | 123,360 | 131,072 | 0.4833 |
| KV int8 + budget | 2.301 | 1.065× | 131,072 | 131,072 | 0.4500 |
| KV int4 + budget | 2.246 | 1.040× | 127,808 | 131,072 | 0.4650 |
