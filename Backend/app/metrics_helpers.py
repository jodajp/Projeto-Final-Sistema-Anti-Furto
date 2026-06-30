from typing import Optional
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.metrica import MetricaNodeModel
from app.models.zona import ZonaModel


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

        # Agrupamos as pessoas detetadas por hora em memória
        hourly_data = {h: [] for h in range(24)}
        for row in rows:
            active_time = row.data_recebida or row.timestamp
            if active_time is None:
                continue
            hour_index = active_time.hour
            hourly_data[hour_index].append(row.pessoas_detetadas or 0)

        # Calculamos o pico (máximo) para cada hora
        hours = [0] * 24
        for h in range(24):
            if hourly_data[h]:
                hours[h] = max(hourly_data[h]) # <-- Obtém o pico daquela hora

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

def build_zone_stats_by_day(day: Optional[str], db: Session):
    try:
        # 1. Definir o intervalo de tempo
        if day:
            target_date = datetime.strptime(day, "%Y-%m-%d").date()
        else:
            target_date = datetime.now().date()

        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        # 2. Query agrupada por nome da zona
        resultados = db.query(
            ZonaModel.zone_name,
            func.count(ZonaModel.id).label('total_acessos')
        ).filter(
            ZonaModel.timestamp >= start,
            ZonaModel.timestamp <= end
        ).group_by(
            ZonaModel.zone_name
        ).order_by(
            func.count(ZonaModel.id).desc()
        ).all()

        # 3. Formatar os dados para o Frontend
        labels = []
        counts = []
        for row in resultados:
            labels.append(row.zone_name)
            counts.append(row.total_acessos)

        return {
            "day": target_date.isoformat(),
            "labels": labels,
            "counts": counts,
            "total_events": sum(counts),
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))