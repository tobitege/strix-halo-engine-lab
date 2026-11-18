The host wire headers and the dimension/state layout headers in
`src/glm/rtxpf/import`, and the CPU receiver/replay fixtures in
`tests/glm_native`, are adopted from this project's GLM node import branch at
revision `298ede2f5241d870fc50daa5c7b80a54a052c129`. They retain the project's
license. They are kept under the RTX tree so the node engine's sources are not
changed. The production sender uses those wire types directly; it does not
implement a different protocol or change the receiver's accepted identities.
