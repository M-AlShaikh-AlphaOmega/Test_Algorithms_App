from enum import Enum


class ActivityType(Enum):
    WALKING = "walking"
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    UNKNOWN = "unknown"


ACTIVITY_LABELS = {
    0: ActivityType.WALKING,
    1: ActivityType.STANDING,
    2: ActivityType.SITTING,
    3: ActivityType.LYING,
}


def get_activity_name(label: int) -> str:
    return ACTIVITY_LABELS.get(label, ActivityType.UNKNOWN).value
