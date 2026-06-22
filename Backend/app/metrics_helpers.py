from typing import Optional
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.metrica import MetricaNodeModel


def build_metrics_by_day(day: Optional[str], db: Session):
    try:
        if day:
            target_date = datetime.strptime(day, "%Y-%m-%d").date()
        else:
            target_date = datetime.now().date()

        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        timestamp_field = func.coalesce(MetricaNodeModel.data_recebida, MetricaNodeModel.timestamp)
        rows = db.query(MetricaNodeModel).filter(timestamp_field >= start).filter(timestamp_field <= end).all()

        hours = [0] * 24
        for row in rows:
            active_time = row.data_recebida or row.timestamp
            if active_time is None:
                continue
            hour_index = active_time.hour
            hours[hour_index] += int(row.pessoas_detetadas or 0)

        return {
            "day": target_date.isoformat(),
            "hours": [f"{h:02d}:00" for h in range(24)],
            "counts": hours,
            "max_count": max(hours) if hours else 0,
            "rows_queried": len(rows),
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
