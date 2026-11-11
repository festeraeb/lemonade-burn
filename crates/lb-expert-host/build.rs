// The RDMA transport's verbs shim (feature `rdma`): the inline verbs calls as plain C functions,
// linked with libibverbs. The default build has no native dependency.
fn main() {
    println!("cargo:rerun-if-changed=csrc/w2w_rdma.c");
    #[cfg(feature = "rdma")]
    {
        cc::Build::new().file("csrc/w2w_rdma.c").warnings(true).opt_level(2).compile("w2w_rdma_shim");
        println!("cargo:rustc-link-lib=ibverbs");
    }
}
