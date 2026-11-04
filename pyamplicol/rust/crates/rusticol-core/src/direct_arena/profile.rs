// SPDX-License-Identifier: 0BSD

//! Lane-neutral direct-call and forbidden-materialization accounting.

use crate::{RusticolError, RusticolResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectArenaTrafficKind {
    PacketInput,
    PacketOutput,
    Gather,
    Scatter,
    Remap,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectArenaTrafficCounters {
    pub calls: u64,
    pub rows: u64,
    pub points: u64,
    pub packet_input_bytes: u64,
    pub packet_output_bytes: u64,
    pub gather_bytes: u64,
    pub scatter_bytes: u64,
    pub remap_bytes: u64,
}

impl DirectArenaTrafficCounters {
    pub fn record_call(&mut self, rows: u32, points: u32) {
        self.calls = self.calls.saturating_add(1);
        self.rows = self.rows.saturating_add(u64::from(rows));
        self.points = self.points.saturating_add(u64::from(points));
    }

    pub fn record_traffic(&mut self, kind: DirectArenaTrafficKind, bytes: u64) {
        let counter = match kind {
            DirectArenaTrafficKind::PacketInput => &mut self.packet_input_bytes,
            DirectArenaTrafficKind::PacketOutput => &mut self.packet_output_bytes,
            DirectArenaTrafficKind::Gather => &mut self.gather_bytes,
            DirectArenaTrafficKind::Scatter => &mut self.scatter_bytes,
            DirectArenaTrafficKind::Remap => &mut self.remap_bytes,
        };
        *counter = counter.saturating_add(bytes);
    }

    pub fn validate_direct(&self) -> RusticolResult<()> {
        if self.packet_input_bytes
            | self.packet_output_bytes
            | self.gather_bytes
            | self.scatter_bytes
            | self.remap_bytes
            != 0
        {
            return Err(RusticolError::integrity(
                "Direct-Arena execution recorded forbidden packet/gather/scatter/remap traffic",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_traffic_fails_closed() {
        let mut counters = DirectArenaTrafficCounters::default();
        counters.record_call(7, 128);
        assert!(counters.validate_direct().is_ok());
        counters.record_traffic(DirectArenaTrafficKind::Gather, 8);
        assert!(counters.validate_direct().is_err());
    }
}
