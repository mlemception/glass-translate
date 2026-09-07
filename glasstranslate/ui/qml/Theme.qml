pragma Singleton
pragma ComponentBehavior: Bound
import QtQuick

/*!
    Theme - design tokens, mode derivation and geometry helpers for the glass UI
    (docs/GLASS_DESIGN.md §1.3, §2.3).

    Reads the context objects `appearance` (Appearance QObject) and `bridge`
    (ControlBridge) when they exist; without them it falls back to glass mode,
    light polarity, no reduced motion and text scale 1, so the components also
    work in the dev lab and in isolated tests.

    Accent inventory (nothing else may use `accent`): Start/Stop primary fill
    22 % (`fillPrimary`), toggle on-track, slider fill, 6-px running dot, focus
    ring (`focusColor`), download progress bar.  There is no warn colour.
*/
QtObject {
    id: theme

    // ------------------------------------------------------------ context objects
    // Context properties are unqualified by nature; the typeof guards keep Theme
    // usable when they are absent.
    // qmllint disable unqualified
    readonly property bool hasAppearance: typeof appearance !== "undefined" && appearance !== null
    readonly property bool hasBridge: typeof bridge !== "undefined" && bridge !== null
    readonly property bool transparency: hasAppearance ? appearance.transparency : true
    readonly property bool composition: hasAppearance ? appearance.composition : true
    readonly property bool highContrast: hasAppearance ? appearance.highContrast : false
    readonly property bool darkMode: hasAppearance ? appearance.darkMode : false
    readonly property bool reduceMotion: hasAppearance ? appearance.reduceMotion : false
    readonly property real textScale: hasAppearance ? appearance.textScale : 1.0
    readonly property int bridgePolarity: hasBridge ? bridge.inkPolarity : (darkMode ? 0 : 1)
    // qmllint enable unqualified

    // ------------------------------------------------------------ mode
    //  glass: live backdrop, refraction, springs, no borders
    //  solid: opaque frosted fill, springs unless reduceMotion
    //  hc:    system palette, 2 px borders, no shadow, no specular
    readonly property string mode: highContrast ? "hc" : (!transparency || !composition) ? "solid" : "glass"
    readonly property bool glassAllowed: mode === "glass"
    readonly property SystemPalette sysPal: SystemPalette { colorGroup: SystemPalette.Active }

    // ------------------------------------------------------------ polarity / ink
    // 0 = dark material + light ink, 1 = light material + dark ink.  The backdrop
    // polarity is only trusted in glass mode (the grabber is paused otherwise).
    readonly property real inkTarget: mode === "glass" ? bridgePolarity : (darkMode ? 0 : 1)
    property real inkValue: inkTarget
    Behavior on inkValue { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }

    readonly property color inkLight: Qt.rgba(0.96, 0.97, 1.00, 1)
    readonly property color inkDark: Qt.rgba(0.10, 0.11, 0.13, 1)
    property color ink: mode === "hc" ? sysPal.windowText : (inkTarget > 0.5 ? inkDark : inkLight)
    Behavior on ink { ColorAnimation { duration: 180 } }
    property color inkSecondary: mode === "hc" ? sysPal.windowText
                                              : (inkTarget > 0.5 ? Qt.rgba(0.10, 0.11, 0.13, 0.62)
                                                                 : Qt.rgba(0.96, 0.97, 1.00, 0.66))
    Behavior on inkSecondary { ColorAnimation { duration: 180 } }

    // ------------------------------------------------------------ colours
    readonly property color accent: mode === "hc" ? sysPal.highlight : "#3A86FF"
    readonly property color tintColor: Qt.rgba(0.90, 0.96, 1.00, 1)          // alpha 1: used as an RGB factor
    readonly property color fill: mode === "hc" ? sysPal.window
                                : (darkMode ? Qt.rgba(0.14, 0.15, 0.18, 1) : Qt.rgba(0.94, 0.95, 0.97, 1))
    readonly property color fillControl: mode === "hc" ? sysPal.button : Qt.rgba(1, 1, 1, 0.10 + 0.45 * inkValue)
    readonly property color fillCard: mode === "hc" ? sysPal.base : Qt.rgba(1, 1, 1, 0.07 + 0.35 * inkValue)
    readonly property color fillTrack: mode === "hc" ? sysPal.base
                                     : Qt.rgba(1 - inkValue, 1 - inkValue, 1 - inkValue, 0.10 - 0.03 * inkValue)
    readonly property color fillPill: mode === "hc" ? sysPal.button : Qt.rgba(1, 1, 1, 0.25 + 0.65 * inkValue)
    readonly property color fillPrimary: mode === "hc" ? sysPal.highlight
                                       : over(Qt.rgba(accent.r, accent.g, accent.b, 0.22), fillControl)
    readonly property color focusColor: mode === "hc" ? sysPal.highlight : Qt.rgba(accent.r, accent.g, accent.b, 0.60)
    readonly property color borderColor: mode === "hc" ? sysPal.windowText : Qt.rgba(1, 1, 1, 1)

    // ------------------------------------------------------------ metrics (heights are minimums)
    readonly property real radiusWindow: 22
    readonly property real radiusCard: 14
    readonly property real radiusControl: 10
    readonly property real pad: 14
    readonly property real padY: 6
    readonly property real gap: 10
    readonly property real controlH: 34
    readonly property real titleH: 40
    readonly property real stripH: 44

    // ------------------------------------------------------------ fonts (points, scaled by Windows text size)
    readonly property string family: Qt.fontFamilies().indexOf("Segoe UI Variable Text") >= 0 ? "Segoe UI Variable Text" : "Segoe UI"
    readonly property string mono: Qt.fontFamilies().indexOf("Cascadia Mono") >= 0 ? "Cascadia Mono" : "Consolas"
    readonly property real fontBody: Application.font.pointSize * textScale
    readonly property real fontCaption: fontBody - 1
    readonly property real fontTitle: fontBody + 2

    // ------------------------------------------------------------ motion
    // Every spring goes through these; under reduceMotion use a NumberAnimation of
    // `motionDuration` ms OutCubic instead (Behavior.animation is assignable).
    component Spring: QtObject {
        required property real spring
        required property real damping
    }
    readonly property Spring springPress: Spring { spring: 5.0; damping: 0.32 }
    readonly property Spring springGrab: Spring { spring: 5.0; damping: 0.38 }
    readonly property Spring springSwell: Spring { spring: 3.6; damping: 0.30 }
    readonly property Spring springLead: Spring { spring: 4.6; damping: 0.36 }
    readonly property Spring springTrail: Spring { spring: 3.0; damping: 0.30 }
    readonly property int motionDuration: 120
    readonly property int fade: 160
    readonly property int pressInMs: 90
    readonly property real pressScale: reduceMotion ? 1.0 : 0.965
    readonly property real grabScale: reduceMotion ? 1.0 : 1.28

    // ------------------------------------------------------------ pointer (root coordinates)
    // Main.qml binds `rootSize` to the root item and `pointer` to its HoverHandler.
    // At `pointerRest` both specular lobes are zero for every item (ambient only).
    property size rootSize: Qt.size(832, 640)
    // the slab rect (root = window, slab inset 36 / 28 / 36 / 52): popups are bounded by it
    readonly property rect slabRect: Qt.rect(36, 28, rootSize.width - 72, rootSize.height - 80)
    readonly property point pointerRest: Qt.point(-1.5 * rootSize.width, -1.5 * rootSize.height)
    property point pointer: pointerRest

    // ------------------------------------------------------------ helpers
    function radiusPill(h) { return h / 2 }
    function concentric(outerRadius, inset) { return Math.max(4, outerRadius - inset) }

    // Source-over of two straight (non-premultiplied) colours.
    function over(top, bottom) {
        var a = top.a + bottom.a * (1 - top.a)
        if (a <= 0)
            return Qt.rgba(0, 0, 0, 0)
        return Qt.rgba((top.r * top.a + bottom.r * bottom.a * (1 - top.a)) / a,
                       (top.g * top.a + bottom.g * bottom.a * (1 - top.a)) / a,
                       (top.b * top.a + bottom.b * bottom.a * (1 - top.a)) / a, a)
    }

    // Geometry helpers - the only sanctioned way to compute source rects and pointer
    // positions.  Reading every ancestor's x/y (Flickable contentItem, page slides,
    // popups) makes QML capture them as binding dependencies; mapToItem does not.
    // `scale` is deliberately ignored (a press-compress scales the lens as a whole).
    function scenePos(it) {
        var x = 0, y = 0
        while (it) {
            x += it.x
            y += it.y
            it = it.parent
        }
        return Qt.point(x, y)
    }
    function rectIn(item, src) {
        var a = scenePos(item), b = scenePos(src)
        return Qt.rect((a.x - b.x) / src.width, (a.y - b.y) / src.height, item.width / src.width, item.height / src.height)
    }
    function pointerIn(item) {
        var a = scenePos(item)
        return Qt.point(pointer.x - a.x - item.width / 2, pointer.y - a.y - item.height / 2)
    }
}
