// SPDX-License-Identifier: 0BSD

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use rusticol_core::direct_arena::{
    DirectArenaTrafficCounters, DirectArenaView, DirectArenaWorkspace, DirectPlaneShape,
    DirectPointTile,
};

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn count_allocation(bytes: usize) {
    if TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
        let _ = ALLOCATED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
    }
}

fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATION_COUNT.with(|count| count.set(0));
    ALLOCATED_BYTES.with(|total| total.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = operation();
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    (
        result,
        ALLOCATION_COUNT.with(Cell::get),
        ALLOCATED_BYTES.with(Cell::get),
    )
}

fn write_split(
    values_re: *mut f64,
    values_im: *mut f64,
    shape: DirectPlaneShape,
    component: u32,
    point: u32,
    value_re: f64,
    value_im: f64,
) {
    let index = shape
        .checked_scalar_index(component, point, "test split plane")
        .unwrap();
    unsafe {
        values_re.add(index).write(value_re);
        values_im.add(index).write(value_im);
    }
}

fn read_split(
    values_re: *const f64,
    values_im: *const f64,
    shape: DirectPlaneShape,
    component: u32,
    point: u32,
) -> (f64, f64) {
    let index = shape
        .checked_scalar_index(component, point, "test split plane")
        .unwrap();
    unsafe { (values_re.add(index).read(), values_im.add(index).read()) }
}

fn write_current(view: DirectArenaView, component: u32, point: u32, re: f64, im: f64) {
    write_split(
        view.current_re,
        view.current_im,
        view.current_shape().unwrap(),
        component,
        point,
        re,
        im,
    );
}

fn read_current(view: DirectArenaView, component: u32, point: u32) -> (f64, f64) {
    read_split(
        view.current_re,
        view.current_im,
        view.current_shape().unwrap(),
        component,
        point,
    )
}

fn write_amplitude(view: DirectArenaView, component: u32, point: u32, re: f64, im: f64) {
    write_split(
        view.amplitude_re,
        view.amplitude_im,
        view.amplitude_shape().unwrap(),
        component,
        point,
        re,
        im,
    );
}

fn read_amplitude(view: DirectArenaView, component: u32, point: u32) -> (f64, f64) {
    read_split(
        view.amplitude_re,
        view.amplitude_im,
        view.amplitude_shape().unwrap(),
        component,
        point,
    )
}

#[test]
fn every_physical_plane_base_is_cache_aligned_for_awkward_capacities() {
    for (tile_capacity, expected_stride) in [(1, 8), (7, 8), (127, 128), (129, 136)] {
        let mut workspace = DirectArenaWorkspace::new(3, 2, tile_capacity).unwrap();
        assert_eq!(workspace.tile_capacity(), tile_capacity);
        assert_eq!(workspace.point_stride(), expected_stride);
        workspace.begin_tile(tile_capacity).unwrap();
        let view = workspace.view().unwrap();
        for component in 0..3_usize {
            let offset = component * expected_stride as usize;
            assert_eq!(unsafe { view.current_re.add(offset) } as usize % 64, 0);
            assert_eq!(unsafe { view.current_im.add(offset) } as usize % 64, 0);
        }
        for component in 0..2_usize {
            let offset = component * expected_stride as usize;
            assert_eq!(unsafe { view.amplitude_re.add(offset) } as usize % 64, 0);
            assert_eq!(unsafe { view.amplitude_im.add(offset) } as usize % 64, 0);
        }
    }
}

#[test]
fn begin_tile_preserves_storage_and_selective_clear_skips_inactive_tails() {
    let mut workspace = DirectArenaWorkspace::new(3, 2, 129).unwrap();
    workspace.begin_tile(129).unwrap();
    let initial = workspace.view().unwrap();
    assert_eq!(initial.point_stride, 136);
    for component in 0..3 {
        for point in 0..initial.point_stride {
            let value = 10_000.0 + f64::from(component * 1000 + point);
            write_current(initial, component, point, value, -value);
        }
    }
    for component in 0..2 {
        for point in 0..initial.point_stride {
            let value = 20_000.0 + f64::from(component * 1000 + point);
            write_amplitude(initial, component, point, value, -value);
        }
    }

    workspace.begin_tile(127).unwrap();
    workspace.clear_current_active(1, 1).unwrap();
    workspace.clear_amplitude_active(0, 1).unwrap();
    let view = workspace.view().unwrap();
    assert_eq!(view.current_re, initial.current_re);
    assert_eq!(view.amplitude_re, initial.amplitude_re);
    assert_eq!(view.point_stride, initial.point_stride);
    assert_eq!(read_current(view, 1, 126), (0.0, 0.0));
    assert_eq!(read_current(view, 1, 127), (11_127.0, -11_127.0));
    assert_eq!(read_current(view, 0, 0), (10_000.0, -10_000.0));
    assert_eq!(read_amplitude(view, 0, 126), (0.0, 0.0));
    assert_eq!(read_amplitude(view, 0, 127), (20_127.0, -20_127.0));
    assert_eq!(read_amplitude(view, 1, 0), (21_000.0, -21_000.0));
}

#[test]
fn adapter_uses_global_tile_offsets_but_local_workspace_indices() {
    const POINT_COUNT: u32 = 1025;
    const SENTINEL: f64 = -9_876_543.25;
    let input_a = (0..POINT_COUNT)
        .map(|point| f64::from(point) + 0.25)
        .collect::<Vec<_>>();
    let input_b = (0..POINT_COUNT)
        .map(|point| f64::from(point) * -0.5 - 3.0)
        .collect::<Vec<_>>();
    let mut output = vec![SENTINEL; POINT_COUNT as usize + 2];
    let mut workspace = DirectArenaWorkspace::new(2, 1, 129).unwrap();
    let tiles = workspace.point_tiles(POINT_COUNT, 128).unwrap();
    let mut final_tile = None;

    for tile in tiles {
        workspace.begin_tile(tile.point_count).unwrap();
        let view = workspace.view().unwrap();
        for local in 0..tile.point_count {
            let global = tile.point_start + local;
            write_current(view, 0, local, input_a[global as usize], 0.0);
            write_current(view, 1, local, input_b[global as usize], 0.0);
        }
        for local in 0..tile.point_count {
            let left = read_current(view, 0, local).0;
            let right = read_current(view, 1, local).0;
            write_amplitude(view, 0, local, left + 2.0 * right, 0.0);
            output[(tile.point_start + local) as usize + 1] = read_amplitude(view, 0, local).0;
        }
        final_tile = Some(tile);
    }
    assert_eq!(
        final_tile,
        Some(DirectPointTile {
            point_start: 1024,
            point_count: 1,
        })
    );
    assert_eq!(output[0], SENTINEL);
    assert_eq!(output[POINT_COUNT as usize + 1], SENTINEL);
    for point in 0..POINT_COUNT as usize {
        assert_eq!(output[point + 1], input_a[point] + 2.0 * input_b[point]);
    }
}

#[test]
fn warmed_begin_view_selective_clear_and_tile_reuse_allocate_zero_bytes() {
    let mut workspace = DirectArenaWorkspace::new(97, 13, 129).unwrap();
    workspace.begin_tile(129).unwrap();
    workspace.clear_current_active(0, 97).unwrap();
    workspace.clear_amplitude_active(0, 13).unwrap();
    black_box(workspace.view().unwrap());
    let allocation_counters = workspace.allocation_counters();
    let traffic_counters = DirectArenaTrafficCounters::default();

    let (tail, allocation_count, allocated_bytes) = count_allocations(|| {
        let mut tail = DirectPointTile {
            point_start: 0,
            point_count: 0,
        };
        for points in [127, 129, 128, 1].into_iter().cycle().take(64) {
            workspace.begin_tile(points).unwrap();
            workspace.clear_current_active(3, 71).unwrap();
            workspace.clear_amplitude_active(2, 7).unwrap();
            black_box(workspace.view().unwrap());
            for tile in workspace.point_tiles(1025, points).unwrap() {
                tail = black_box(tile);
            }
        }
        tail
    });

    assert_eq!(
        tail,
        DirectPointTile {
            point_start: 1024,
            point_count: 1,
        }
    );
    assert_eq!(allocation_count, 0, "warmed workspace path allocated");
    assert_eq!(allocated_bytes, 0, "warmed workspace path allocated bytes");
    assert_eq!(workspace.allocation_counters(), allocation_counters);
    assert_eq!(traffic_counters, DirectArenaTrafficCounters::default());
}
