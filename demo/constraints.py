"""Ràng buộc là dữ liệu; prompt và kiểm tra dùng cùng một nguồn."""

from datetime import date as Date, datetime, time as Time

from pydantic import BaseModel, ConfigDict, Field


class Constraints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: str = Field(default="SGN", pattern=r"^[A-Z]{3}$")
    destination: str = Field(default="DAD", pattern=r"^[A-Z]{3}$")
    date: Date = Date(2026, 10, 7)
    depart_before: Time = Time(12, 0)
    max_price: int = Field(default=2_000_000, gt=0)

    def to_prompt(self) -> str:
        return (
            f"Book exactly one ticket {self.origin} -> {self.destination} "
            f"on {self.date.isoformat()}, departing before "
            f"{self.depart_before.strftime('%H:%M')}, at most {self.max_price:,} VND."
        )

    def violations(self, flight: dict) -> list[str]:
        """Kiểm cả tuyến, ngày, giờ, tiền tệ và giá; dữ liệu lỗi không hợp lệ."""
        try:
            departure = datetime.fromisoformat(flight["depart"])
            price = flight["price"]
            errors = []
            if (flight["origin"], flight["destination"]) != (self.origin, self.destination):
                errors.append("Sai tuyến bay")
            if departure.date() != self.date:
                errors.append("Sai ngày bay")
            if departure.time() >= self.depart_before:
                errors.append("Khởi hành quá muộn")
            if type(price) is not int or not 0 < price <= self.max_price:
                errors.append("Giá không hợp lệ hoặc vượt ngân sách")
            if flight["currency"] != "VND":
                errors.append("Sai tiền tệ")
            return errors
        except (KeyError, TypeError, ValueError):
            return ["Dữ liệu chuyến bay thiếu hoặc sai định dạng"]

    def is_ok(self, flight: dict) -> bool:
        return not self.violations(flight)
