# Migration Guide: v0.5.0 → v0.6.0-alpha

This guide helps you migrate from god-graph v0.5.0 to v0.6.0-alpha.

## Summary of Breaking Changes

| Change | Severity | Migration Effort |
|--------|----------|------------------|
| Module renaming `impl_` → `graph_impl` | 🔴 High | Low (find & replace) |
| Backend trait moved to VGI | 🟡 Medium | Low (update imports) |
| Feature flag simplification | 🟡 Medium | Low (update Cargo.toml) |

---

## Step 1: Update Cargo.toml

### Remove Deprecated Features

```toml
# ❌ Old (v0.5.0)
[dependencies]
god-graph = { version = "0.5.0", features = ["tensor-sparse", "rand_chacha"] }

# ✅ New (v0.6.0-alpha)
[dependencies]
god-graph = { version = "0.6.0-alpha", features = ["tensor", "rand"] }
```

### Feature Mapping Table

| Old Feature | New Feature | Notes |
|-------------|-------------|-------|
| `tensor-sparse` | `tensor` | Now includes sparse formats by default |
| `rand_chacha` | `rand` | Simplified random number generation |
| `prefetch` | _(removed)_ | Always enabled with `unstable` feature |

---

## Step 2: Update Imports

### Module Renaming

```rust
// ❌ Old (v0.5.0)
use god_graph::graph::impl_::Graph;
use god_graph::graph::impl_::NodeIndex;

// ✅ New (v0.6.0-alpha)
use god_graph::graph::graph_impl::Graph;
use god_graph::graph::graph_impl::NodeIndex;
```

**Find & Replace Pattern:**
```bash
# In your project root
find . -name "*.rs" -type f -exec sed -i 's/graph::impl_::/graph::graph_impl::/g' {} \;
```

### Backend Trait Migration

```rust
// ❌ Old (v0.5.0)
use god_graph::backend::{Backend, BackendConfig, BackendType};

// ✅ New (v0.6.0-alpha)
use god_graph::vgi::{Backend, BackendConfig, BackendType};
```

**Note:** `backend::traits` still re-exports these types for backward compatibility, but this is deprecated and will be removed in v0.7.0.

---

## Step 3: Update Feature Flags in Code

```rust
// ❌ Old (v0.5.0)
#[cfg(feature = "tensor-sparse")]
mod sparse_tensor_ops {
    // ...
}

// ✅ New (v0.6.0-alpha)
#[cfg(feature = "tensor")]
mod sparse_tensor_ops {
    // ...
}
```

**Find & Replace Pattern:**
```bash
find . -name "*.rs" -type f -exec sed -i 's/feature = "tensor-sparse"/feature = "tensor"/g' {} \;
```

---

## Step 4: Verify Compilation

After making the changes:

```bash
# Clean build
cargo clean
cargo build --features "parallel,simd,tensor"

# Run tests
cargo test --features "parallel,simd,tensor"
```

---

## Common Issues

### Issue 1: "could not find `impl_` in `graph`"

**Solution:** Update imports as shown in Step 2.

### Issue 2: "use of undeclared type `Backend`"

**Solution:** Change import from `backend::traits` to `vgi::traits`.

### Issue 3: Feature `tensor-sparse` is not recognized

**Solution:** Replace `tensor-sparse` with `tensor` in both `Cargo.toml` and source code.

---

## New Features in v0.6.0-alpha

After migrating, you can take advantage of:

1. **VGI Architecture**: Cleaner separation between interface and implementation
2. **Simplified Features**: Fewer feature flags to manage
3. **Better Module Names**: `graph_impl` is more descriptive than `impl_`

---

## Need Help?

- Check the [CHANGELOG](CHANGELOG.md) for detailed changes
- Report issues on [GitHub](https://github.com/silverenternal/god-graph/issues)
- Join discussions in [Discussions](https://github.com/silverenternal/god-graph/discussions)
