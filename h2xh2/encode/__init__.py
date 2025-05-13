# Copyright 2025 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoding utilities.

It is essentially working as follows. Each module has `encode` and `interpret`
functions, and you can the encoding and interpretation as
```
        :
        :
    encoded_circuit = encode(
        logical_circuit,
        EncodingOptions(...),
    )
        :
    # (execute jobs)
        :
    logical_result = interpret(
        encoded_result,
        InterpretOptions(...),
    )
```
"""

__all__ = [
    "plain",
    "steane",
]
