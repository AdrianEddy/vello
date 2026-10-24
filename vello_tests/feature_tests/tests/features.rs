// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compare the common probe scene under the selected shader feature combination.

#[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
mod support;

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn feature_tests_probe_reference() {
    let elements = support::elements();
    let actual = support::render_cpu(&elements);
    support::check_snapshot(actual, "cpu_reference", 1, true);
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn feature_tests_probe_wgpu() {
    let elements = support::elements();
    let actual = support::wgpu::render(&elements);
    support::check_snapshot(actual, "wgpu", 3, false);
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
#[wasm_bindgen_test::wasm_bindgen_test]
fn feature_tests_probe_webgl() {
    let elements = support::elements();
    let actual = support::webgl::render(&elements);
    support::check_snapshot(actual, "webgl", 3, false);
}

/// Like image uploads, `render_to_image` requires `full_image_support`.
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn feature_tests_render_to_image_wgpu() {
    assert_eq!(
        support::wgpu::render_to_image_succeeds(),
        cfg!(feature = "full_image_support")
    );
}

/// Like image uploads, `render_to_image` requires `full_image_support`.
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
#[wasm_bindgen_test::wasm_bindgen_test]
fn feature_tests_render_to_image_webgl() {
    let result = support::webgl::render_to_image();
    if cfg!(feature = "full_image_support") {
        assert!(result.is_ok(), "rendering into an image must succeed");
    } else {
        assert!(
            matches!(result, Err(vello_gpu::WebGlError::FullImageSupportDisabled)),
            "rendering into an image must report the missing feature"
        );
    }
}
