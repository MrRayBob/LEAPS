from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fits_inventory import validate_coordinates
from .models import LEAPSError, StageID


@dataclass(slots=True)
class PlanetParameters:
    name: str
    ra: str
    dec: str
    period: float
    mid_time: float
    rp_over_rs: float
    sma_over_rs: float
    inclination: float
    eccentricity: float
    periastron: float
    metallicity: float
    temperature: float
    logg: float
    source: str
    source_date: str = ""


class PlanetCatalogResolver:
    def __init__(self, nasa_snapshot: str | Path | None = None) -> None:
        self.nasa_snapshot = Path(nasa_snapshot).expanduser() if nasa_snapshot else None

    def resolve(self, ra: str, dec: str, name: str = "") -> PlanetParameters:
        candidates = self.resolve_candidates(ra, dec, name)
        if candidates:
            return candidates[0]
        raise LEAPSError(
            "PLANET_NOT_FOUND",
            "No planet parameters were found",
            "The target coordinates are valid, but neither ExoClock nor the offline NASA snapshot contains a match.",
            ["Check the target coordinates", "Update offline data and retry"],
            stage=StageID.FITTING,
        )

    def resolve_candidates(self, ra: str, dec: str, name: str = "") -> list[PlanetParameters]:
        """Return all catalogued planets at the target, ordered for easy selection."""
        ra, dec = validate_coordinates(ra, dec)
        exoclock_results = self._candidates_from_exoclock(ra, dec, name)
        if exoclock_results:
            return exoclock_results
        return self._candidates_from_nasa(name, ra, dec)

    @staticmethod
    def _from_exoclock(ra: str, dec: str) -> PlanetParameters | None:
        try:
            import exoclock

            data = exoclock.locate_planet(exoclock.Hours(ra), exoclock.Degrees(dec))
            return PlanetParameters(
                name=data["name"],
                ra=str(data["star"]["ra"]),
                dec=str(data["star"]["dec"]),
                period=float(data["planet"]["ephem_period"]),
                mid_time=float(data["planet"]["ephem_mid_time"]),
                rp_over_rs=float(data["planet"]["rp_over_rs"]),
                sma_over_rs=float(data["planet"]["sma_over_rs"]),
                inclination=float(data["planet"]["inclination"]),
                eccentricity=float(data["planet"]["eccentricity"]),
                periastron=float(data["planet"]["periastron"]),
                metallicity=float(data["planet"]["meta"]),
                temperature=float(data["planet"]["teff"]),
                logg=float(data["planet"]["logg"]),
                source="ExoClock",
            )
        except Exception:
            return None

    @classmethod
    def _candidates_from_exoclock(
        cls, ra: str, dec: str, name: str
    ) -> list[PlanetParameters]:
        try:
            import astropy.units as units
            import exoclock
            from astropy.coordinates import SkyCoord

            target = SkyCoord(ra, dec, unit=(units.hourangle, units.deg), frame="icrs")
            matches: list[tuple[float, PlanetParameters]] = []
            for planet_name in exoclock.get_all_planets():
                data = exoclock.get_planet(planet_name)
                coordinate = SkyCoord(
                    float(data["star"]["ra_deg"]),
                    float(data["star"]["dec_deg"]),
                    unit=(units.deg, units.deg),
                )
                separation = float(target.separation(coordinate).deg)
                if separation <= 0.02:
                    matches.append((separation, cls._exoclock_parameters(data)))
            requested = _flat_name(name)
            matches.sort(
                key=lambda item: (
                    0 if requested and _flat_name(item[1].name).startswith(requested) else 1,
                    item[0],
                    item[1].name.casefold(),
                )
            )
            return [parameters for _, parameters in matches]
        except Exception:
            fallback = cls._from_exoclock(ra, dec)
            return [fallback] if fallback else []

    @staticmethod
    def _exoclock_parameters(data: dict[str, Any]) -> PlanetParameters:
        return PlanetParameters(
            name=data["name"],
            ra=str(data["star"]["ra"]),
            dec=str(data["star"]["dec"]),
            period=float(data["planet"]["ephem_period"]),
            mid_time=float(data["planet"]["ephem_mid_time"]),
            rp_over_rs=float(data["planet"]["rp_over_rs"]),
            sma_over_rs=float(data["planet"]["sma_over_rs"]),
            inclination=float(data["planet"]["inclination"]),
            eccentricity=float(data["planet"]["eccentricity"]),
            periastron=float(data["planet"]["periastron"]),
            metallicity=float(data["planet"]["meta"]),
            temperature=float(data["planet"]["teff"]),
            logg=float(data["planet"]["logg"]),
            source="ExoClock",
        )

    def _from_nasa(self, name: str, ra: str, dec: str) -> PlanetParameters | None:
        candidates = self._candidates_from_nasa(name, ra, dec)
        return candidates[0] if candidates else None

    def _candidates_from_nasa(self, name: str, ra: str, dec: str) -> list[PlanetParameters]:
        if not self.nasa_snapshot or not self.nasa_snapshot.exists():
            return []
        payload: dict[str, Any] | list[dict[str, Any]] = json.loads(
            self.nasa_snapshot.read_text(encoding="utf-8")
        )
        candidates = payload.get("planets", []) if isinstance(payload, dict) else payload
        normalized = _flat_name(name)
        matches: list[PlanetParameters] = []
        for record in candidates:
            record_name = str(record.get("pl_name", ""))
            comparable = _flat_name(record_name)
            host = _flat_name(record.get("hostname", ""))
            if normalized and not (comparable.startswith(normalized) or host == normalized):
                continue
            try:
                matches.append(PlanetParameters(
                    name=record_name,
                    ra=str(record.get("ra_str") or ra),
                    dec=str(record.get("dec_str") or dec),
                    period=float(record["pl_orbper"]),
                    mid_time=float(record["pl_tranmid"]),
                    rp_over_rs=float(record["pl_ratror"]),
                    sma_over_rs=float(record["pl_ratdor"]),
                    inclination=float(record["pl_orbincl"]),
                    eccentricity=float(record.get("pl_orbeccen") or 0),
                    periastron=float(record.get("pl_orblper") or 0),
                    metallicity=float(record.get("st_met") or 0),
                    temperature=float(record["st_teff"]),
                    logg=float(record["st_logg"]),
                    source="NASA Exoplanet Archive",
                    source_date=str(payload.get("snapshot_date", "")) if isinstance(payload, dict) else "",
                ))
            except (KeyError, TypeError, ValueError):
                continue
        matches.sort(
            key=lambda parameters: (
                0 if normalized and _flat_name(parameters.name).startswith(normalized) else 1,
                parameters.name.casefold(),
            )
        )
        return matches


def _flat_name(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
