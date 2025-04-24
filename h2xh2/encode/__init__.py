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
