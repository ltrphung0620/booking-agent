"""Hệ thống vé giả, tách riêng dữ liệu của mỗi lần chạy; không gọi mạng."""

from copy import deepcopy
from threading import RLock

from langchain_core.tools import tool

SCENARIOS = ("success", "no_match", "sold_out", "search_error")


class MockFlightService:
    def __init__(self, scenario: str = "success"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.lock = RLock()
        self.bookings: dict[str, dict] = {}
        self.flights = [
            self._flight("VN122", "08:10", 1_850_000),
            self._flight("VN124", "09:20", 1_950_000),
            self._flight("QH118", "15:40", 1_640_000),
        ]
        if scenario == "no_match":
            self.flights = [
                self._flight("VJ604", "08:10", 2_480_000),
                self._flight("QH118", "15:40", 1_640_000),
            ]

    @staticmethod
    def _flight(code: str, departure: str, price: int) -> dict:
        return {
            "flight": code, "origin": "SGN", "destination": "DAD",
            "depart": f"2026-10-07T{departure}:00+07:00",
            "price": price, "currency": "VND", "available": True,
        }

    def find_flight(self, flight: str) -> dict | None:
        return next((f for f in self.flights if f["flight"] == flight), None)

    def search_flights(self, origin: str, destination: str, date: str) -> dict:
        with self.lock:
            if self.scenario == "search_error":
                return {"status": "error", "reason": "Dịch vụ tìm vé bị timeout"}
            found = [f for f in self.flights if f["origin"] == origin
                     and f["destination"] == destination
                     and f["depart"][:10] == date and f["available"]]
            return {"status": "ok", "flights": deepcopy(found)}

    def book_seat(self, flight: str) -> dict:
        with self.lock:
            info = self.find_flight(flight)
            if info is None:
                return {"status": "not_found", "reason": "Không tồn tại chuyến bay"}
            # Giả lập dữ liệu thay đổi SAU khi tìm vé, TRƯỚC khi giữ chỗ.
            if self.scenario == "sold_out" and flight == "VN122":
                info["available"] = False
            if not info["available"]:
                return {"status": "unavailable", "reason": f"{flight} đã hết chỗ"}
            code = f"{flight}-12A"
            if code not in self.bookings:
                self.bookings[code] = {"code": code, "seat": "12A", "paid": False,
                                       **deepcopy(info)}
            return {"status": "ok", **deepcopy(self.bookings[code])}

    def pay(self, code: str) -> dict:
        with self.lock:
            if code not in self.bookings:
                return {"status": "not_found", "reason": "Không tồn tại booking"}
            # Idempotent: gọi lại không tạo giao dịch thanh toán thứ hai.
            self.bookings[code]["paid"] = True
            return {"status": "ok", **deepcopy(self.bookings[code])}

    def get_booking(self, code: str) -> dict:
        with self.lock:
            if code not in self.bookings:
                return {"status": "not_found", "reason": "Không tồn tại booking"}
            return {"status": "ok", **deepcopy(self.bookings[code])}

    def make_tools(self) -> dict:
        """Đăng ký các closure: mỗi agent dùng đúng service của lần chạy đó."""
        @tool
        def search_flights(origin: str, destination: str, date: str) -> dict:
            """Search available flights by airport codes and date (YYYY-MM-DD)."""
            return self.search_flights(origin, destination, date)

        @tool
        def book_seat(flight: str) -> dict:
            """Hold one seat on a flight. Returns a booking code; does not pay."""
            return self.book_seat(flight)

        @tool
        def pay(code: str) -> dict:
            """Pay a held booking by code. All payments in this demo are simulated."""
            return self.pay(code)

        @tool
        def get_booking(code: str) -> dict:
            """Read an existing booking to check its actual paid status."""
            return self.get_booking(code)

        return {t.name: t for t in [search_flights, book_seat, pay, get_booking]}
