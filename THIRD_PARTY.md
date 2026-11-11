# Third-party material

## llama.cpp i-quant tables

`crates/lb-gguf/src/iq_grids.rs` holds lookup tables copied, as data, from llama.cpp
(`ggml/src/ggml-common.h`, commit e14a779719652200cec7ad58dfa4820edc347182): `kmask_iq2xs`,
`ksigns_iq2xs`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid`,
and the `IQ1S_DELTA` / `IQ1M_DELTA` constants. They define the GGUF i-quant formats (the formats
cannot be decoded without them). No llama.cpp code is included; lemonade-burn's decoders are its own.
The 16 IQ4 levels in `crates/lb-gguf/src/types.rs` (`IQ4_LEVELS`) are the same format constant.

Those tables are under the MIT license:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
