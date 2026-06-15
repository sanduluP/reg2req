/**
 * UI-only predicate formatter.
 * Keeps backend values unchanged while showing readable labels to humans.
 */
export function formatPredicateLabel(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";

    const spaced = raw
        .replaceAll("_", " ")
        .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
        .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
        .replace(/\s+/g, " ")
        .trim();

    if (!spaced) return raw;
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
