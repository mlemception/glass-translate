pragma ComponentBehavior: Bound
import QtQuick

/*!
    GlassShadow - analytic Gaussian rounded-box shadow (docs/GLASS_DESIGN.md §2.2,
    §2.3).  Give it the geometry of the surface it shadows (anchors.fill the
    surface or the same x/y/width/height) and draw it BEFORE the surface; it
    paints outside its own bounds on an inner effect inflated by 3*sigma +
    |offset| per axis, and knocks out the box itself so translucent fills are
    never darkened from underneath.  Hidden in high-contrast mode.

        GlassShadow  { anchors.fill: slab; radius: slab.radius; sigma: 12; offsetY: 8 }
        GlassSurface { id: slab; ... }
*/
Item {
    id: root

    property real radius: Theme.radiusWindow
    property real sigma: 12
    property real offsetX: 0
    property real offsetY: 8
    property color color: Qt.rgba(0, 0, 0, 0.32)
    visible: Theme.mode !== "hc"

    readonly property real padX: 3 * sigma + Math.abs(offsetX)
    readonly property real padY: 3 * sigma + Math.abs(offsetY)
    readonly property point offset: Qt.point(offsetX, offsetY)
    readonly property size itemSize: Qt.size(effect.width, effect.height)

    ShaderEffect {
        id: effect
        x: -root.padX
        y: -root.padY
        width: root.width + 2 * root.padX
        height: root.height + 2 * root.padY
        blending: true
        fragmentShader: "shaders/shadow.frag.qsb"

        // uniform block of shadow.frag
        property size itemSize: root.itemSize
        property real radius: root.radius
        property real sigma: root.sigma
        property point offset: root.offset
        property color color: root.color
    }
}
