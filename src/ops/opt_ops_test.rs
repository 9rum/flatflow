// SPDX-License-Identifier: Apache-2.0

use flatbuffers::{FlatBufferBuilder, InvalidFlatbuffer};

use crate::ops::graph::{Graph, Node, SymInt, TensorMetadata};
use crate::ops::graph_generated::{self, root_as_graph};
use crate::ops::operator_generated::Operator;
use crate::ops::scalar_type_generated::ScalarType;
use crate::ops::transform;

#[test]
fn test_transform_with_opt() -> Result<(), InvalidFlatbuffer> {
    // This graph has been generated based on the exported [OPT]:
    // * torch        2.4.0a0+3bcc3cddb5.nv24.7
    // * transformers 4.46.2
    //
    // [OPT]: https://huggingface.co/facebook/opt-13b
    let opt = Graph {
        nodes: vec![
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EMBEDDING,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(50272, 0), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ONES,
                args: vec![],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::FULL,
                args: vec![],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::ARANGE,
                args: vec![],
                meta: TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] }],
                meta: TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] }],
                meta: TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(0, 1), SymInt(1, 0)],
                },
            },
            Node {
                target: Operator::LT_TENSOR,
                args: vec![
                    TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(0, 1), SymInt(1, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::MASKED_FILL_SCALAR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BOOL,
                        shape: vec![SymInt(0, 1), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::RSUB_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::MASKED_FILL_SCALAR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BOOL,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::CUMSUM,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::SUB_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EMBEDDING,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(2050, 0), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT16, shape: vec![SymInt(5120, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::MUL_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::MUL_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::BMM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_SOFTMAX,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::BMM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT16, shape: vec![SymInt(20480, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::RELU,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT16, shape: vec![SymInt(5120, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(50272, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(50272, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(50272, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(50272, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(50272, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(50272, 0)],
                },
            },
        ],
    };

    let mut builder = FlatBufferBuilder::new();
    let mut nodes = Vec::new();
    for node in opt.nodes.iter() {
        let mut args = Vec::new();
        for arg in node.args.iter() {
            let mut shape = Vec::new();
            for int in arg.shape.iter() {
                shape.push(graph_generated::SymInt::new(&[int.0, int.1]));
            }
            let shape = Some(builder.create_vector(shape.as_slice()));

            args.push(graph_generated::TensorMetadata::create(
                &mut builder,
                &graph_generated::TensorMetadataArgs { dtype: arg.dtype, shape },
            ));
        }
        let args = Some(builder.create_vector(args.as_slice()));

        let mut shape = Vec::new();
        for int in node.meta.shape.iter() {
            shape.push(graph_generated::SymInt::new(&[int.0, int.1]));
        }
        let shape = Some(builder.create_vector(shape.as_slice()));

        let meta = Some(graph_generated::TensorMetadata::create(
            &mut builder,
            &graph_generated::TensorMetadataArgs { dtype: node.meta.dtype, shape },
        ));

        nodes.push(graph_generated::Node::create(
            &mut builder,
            &graph_generated::NodeArgs { target: node.target, args, meta },
        ));
    }
    let nodes = Some(builder.create_vector(nodes.as_slice()));

    let graph = graph_generated::Graph::create(&mut builder, &graph_generated::GraphArgs { nodes });
    builder.finish(graph, None);

    let proj = transform(root_as_graph(builder.finished_data())?); // 5261 s0^2 + 246735427 s0
    assert_eq!(proj(0), Ok(0));
    assert_eq!(proj(1), Ok(246740688));
    assert_eq!(proj(1024), Ok(258173635584));
    assert_eq!(proj(2048), Ok(527380387840));

    Ok(())
}
