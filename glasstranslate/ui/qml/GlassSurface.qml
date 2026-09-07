pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window

/*!
    GlassSurface - the Liquid Glass material (docs/GLASS_DESIGN.md §2.2, §2.3).

    Wraps one ShaderEffect running shaders/glass.frag.qsb.  The `variant` string
    selects a preset; every preset value is an ordinary property that an instance
    can override:

      "regular"  slab: rim 34, strength .38, falloff 2, frost 1, frostLift .30, squircle 4.5, specular 1
      "pill"     segmented indicator, slider/toggle thumbs: rim 8, strength .30, falloff 2, frost 0,
                 frostLift .18, squircle 2 (capsules are circular), specular 1, LOCAL sources
      "fill"     buttons, fields, steppers, combo, strip: fill path, fillColor Theme.fillControl,
                 ambient-only specular .5
      "card"     cards and the segmented track: fill path, fillColor Theme.fillCard, specular .35

    Refractive variants ("regular", "pill") sample `sharpSource` / `frostSource`
    (ShaderEffectSources covering `sourceItem`) through `rect`, computed with
    Theme.rectIn (never mapToItem), and bind `pointer`.  Fills and cards run the
    fill path with the pointer at rest; every variant runs the fill path when
    Theme.mode != "glass".  Every scalar uniform is a `property real` (Qt writes
    bool/int as raw bits into float slots).

    Local-source pattern for pills - a thumb lenses the layer of its own control,
    never the slab, and must be a SIBLING of that layer (never its child, which
    would be a recursive source):

        Item { id: trackLayer; ... }                  // track + fill + labels beneath the thumb
        ShaderEffectSource {
            id: trackSrc; sourceItem: trackLayer; hideSource: false; live: true; smooth: true
            textureSize: Qt.size(trackLayer.width * Screen.devicePixelRatio,
                                 trackLayer.height * Screen.devicePixelRatio)
        }
        GlassSurface { variant: "pill"; radius: height / 2; sourceItem: trackLayer; sharpSource: trackSrc }

    The inner ShaderEffect is snapped to device pixels (x, y and both far edges) so
    the 1-device-px specular line lands on one pixel row at 125 % / 150 % scaling.
*/
Item {
    id: root

    /*! "regular" | "fill" | "pill" | "card" */
    property string variant: "regular"
    readonly property bool refractive: variant === "regular" || variant === "pill"

    // ---- shape
    property real radius: variant === "card" ? Theme.radiusCard
                        : variant === "fill" ? Theme.radiusControl
                        : variant === "pill" ? height / 2 : Theme.radiusWindow
    property real squircle: variant === "pill" ? 2 : 4.5

    // ---- refraction
    property real rimWidth: variant === "regular" ? 34 : variant === "pill" ? 8 : 0
    property real strength: variant === "regular" ? 0.38 : variant === "pill" ? 0.30 : 0
    property real falloff: 2.0
    property real frost: variant === "regular" ? 1 : 0
    property real frostLift: variant === "regular" ? 0.30 : variant === "pill" ? 0.18 : 0
    property real coolLean: 0.5
    property color tintColor: Theme.tintColor
    property real ink: Theme.inkValue

    // ---- light
    property real specular: variant === "fill" ? 0.5 : variant === "card" ? 0.35 : 1
    property real shadowInner: variant === "regular" ? 0.08 : 0
    readonly property point pointerAtRest: Qt.point(-1e5, -1e5)
    property point pointer: refractive ? Theme.pointerIn(root) : pointerAtRest

    // ---- sources (refractive variants only)
    /*! The Item whose coordinate space `sharpSource` / `frostSource` cover. */
    property Item sourceItem: null
    property var sharpSource: null
    property var frostSource: null
    property rect rect: sourceItem ? Theme.rectIn(effect, sourceItem) : Qt.rect(0, 0, 1, 1)

    // ---- fill path
    property real solidMode: (refractive && Theme.mode === "glass") ? 0 : 1
    property color fillColor: variant === "card" ? Theme.fillCard
                            : variant === "pill" ? Theme.fillPill
                            : variant === "regular" ? Theme.fill : Theme.fillControl
    property real border: Theme.mode === "hc" ? 2 : 0
    property color borderColor: Theme.borderColor

    // ---- derived uniforms
    readonly property real dpr: Screen.devicePixelRatio > 0 ? Screen.devicePixelRatio : 1
    readonly property size itemSize: Qt.size(effect.width, effect.height)

    // Invariant: strength * falloff <= 0.9, otherwise the outermost pixels show a
    // mirrored copy of the content (a fold).  Clamp and warn.
    readonly property real strengthLimit: 0.9 / Math.max(falloff, 1e-3)
    readonly property real effectiveStrength: {
        if (strength > strengthLimit + 1e-6) {
            console.warn("GlassSurface(" + variant + "): strength " + strength + " * falloff " + falloff
                         + " > 0.9 folds the rim; clamped to " + strengthLimit.toFixed(3))
            return strengthLimit
        }
        return strength
    }

    // Device-pixel snapping of the inner effect (position and far edges).
    readonly property point scenePos: Theme.scenePos(root)
    readonly property real snapX: Math.round(scenePos.x * dpr) / dpr - scenePos.x
    readonly property real snapY: Math.round(scenePos.y * dpr) / dpr - scenePos.y
    readonly property real snapW: Math.round((scenePos.x + width) * dpr) / dpr - scenePos.x - snapX
    readonly property real snapH: Math.round((scenePos.y + height) * dpr) / dpr - scenePos.y - snapY

    ShaderEffect {
        id: effect
        x: root.snapX
        y: root.snapY
        width: root.snapW
        height: root.snapH
        blending: true
        fragmentShader: "shaders/glass.frag.qsb"

        // uniform block of glass.frag, one property per member, same order
        property size itemSize: root.itemSize
        property real dpr: root.dpr
        property real radius: root.radius
        property real squircle: root.squircle
        property real rimWidth: root.rimWidth
        property real strength: root.effectiveStrength
        property real falloff: root.falloff
        property real frost: root.frost
        property real frostLift: root.frostLift
        property real coolLean: root.coolLean
        property color tintColor: Qt.rgba(root.tintColor.r, root.tintColor.g, root.tintColor.b, 1)
        property real ink: root.ink
        property point pointer: root.pointer
        property real specular: Theme.mode === "hc" ? 0 : root.specular
        property real shadowInner: root.shadowInner
        property real solidMode: root.solidMode
        property color fillColor: root.fillColor
        property real border: root.border
        property color borderColor: Qt.rgba(root.borderColor.r, root.borderColor.g, root.borderColor.b, 1)
        property rect rect: root.rect
        property var sharpSource: root.sharpSource
        property var frostSource: root.frostSource
    }
}
