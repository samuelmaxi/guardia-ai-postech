"""Casos sintéticos para demonstração (valores no padrão do dataset; relatos fictícios)."""

EXAMPLES: dict[str, dict] = {
    "Baixo risco": {
        "inputs": {"Age": 22, "SystolicBP": 100, "DiastolicBP": 70, "BS": 6.9, "BodyTemp": 98.0, "HeartRate": 70},
        "relato": "Gestante de 22 semanas, primeira gestação, comparece à consulta de rotina. Refere leve enjoo pela manhã. Sem queixas de dor ou sangramento.",
    },
    "Risco médio": {
        "inputs": {"Age": 29, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.5, "BodyTemp": 100.0, "HeartRate": 76},
        "relato": "Refere febre baixa há dois dias e cansaço. Nega sangramento. Diz que tem dormido pouco.",
    },
    "Alto risco": {
        "inputs": {"Age": 35, "SystolicBP": 140, "DiastolicBP": 100, "BS": 15.0, "BodyTemp": 98.0, "HeartRate": 80},
        "relato": "",
    },
    "Alto risco + relato com sinal de atenção": {
        "inputs": {"Age": 32, "SystolicBP": 140, "DiastolicBP": 95, "BS": 12.0, "BodyTemp": 98.0, "HeartRate": 88},
        "relato": (
            "Gestante de 30 semanas relata dores de cabeça frequentes e inchaço nos pés. Durante a conversa, "
            "diz que o companheiro fica muito nervoso, que na semana passada a empurrou contra a parede e que "
            "ela tem medo de voltar para casa. Pediu para que ele não soubesse do que ela contou."
        ),
    },
}
