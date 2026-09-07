pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window
import QtQuick.Templates as T

/*!
    GlassSegmentedBar - T.TabBar with T.TabButton delegates (docs/GLASS_DESIGN.md
    §2.3).  The track is a capsule "card" fill (Theme.fillTrack) holding the
    labels; the indicator is a "pill" GlassSurface, inset 3 (radius concentric by
    construction), lensing a LOCAL ShaderEffectSource of the track + labels layer.
    Its left and right edges animate separately: the edge in the travel direction
    follows Theme.springLead, the other Theme.springTrail, so the pill stretches
    toward the destination and settles.  Under reduceMotion it jumps with a
    120 ms fade.  In solid / HC the pill is opaque, so the selected label is
    redrawn above it.  The bar is one tab stop; Left/Right move between tabs and
    `activated(index)` fires on user selection.
*/
T.TabBar {
    id: bar

    property var model: ["Translate", "Overlay", "Engines", "Hotkeys"]
    signal activated(int index)

    readonly property real inset: 3
    readonly property int segments: Math.max(1, model.length)
    readonly property real segW: width / segments
    readonly property real dpr: Screen.devicePixelRatio > 0 ? Screen.devicePixelRatio : 1

    // indicator edges (root coordinates of the bar), driven by moveIndicator()
    property real edgeL: inset
    property real edgeR: segW - inset
    property bool goingRight: true

    implicitWidth: 320
    implicitHeight: Math.max(36, fm.height + 2 * Theme.padY)
    padding: 0
    // T.TabBar lays its items out against contentHeight: items without an explicit
    // height are stretched to it and items with one are centred in it.  The default
    // (the ListView's implicit height) is 0, which would put every tab button at
    // y = -h/2 and leave the lower half of the track dead to clicks.
    contentHeight: availableHeight
    spacing: 0
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    focusPolicy: Qt.StrongFocus

    Accessible.role: Accessible.PageTabList
    Accessible.name: "Tabs"

    FontMetrics { id: fm; font: bar.font }

    Keys.onLeftPressed: step(-1)
    Keys.onRightPressed: step(1)

    function step(delta) {
        var i = Math.max(0, Math.min(count - 1, currentIndex + delta))
        if (i !== currentIndex) {
            currentIndex = i
            activated(i)
        }
    }

    function moveIndicator(animate) {
        var l = currentIndex * segW + inset
        var r = (currentIndex + 1) * segW - inset
        var springs = animate && !Theme.reduceMotion
        behL.enabled = springs
        behR.enabled = springs
        goingRight = l > edgeL
        edgeL = l
        edgeR = r
        if (animate && Theme.reduceMotion)
            jumpFade.restart()
    }
    onCurrentIndexChanged: moveIndicator(true)
    onWidthChanged: moveIndicator(false)
    Component.onCompleted: moveIndicator(false)

    // One spring per edge; the lead/trail parameters follow the travel direction
    // (a Behavior's animation object cannot be swapped while it is running).
    Behavior on edgeL {
        id: behL
        SpringAnimation {
            spring: bar.goingRight ? Theme.springTrail.spring : Theme.springLead.spring
            damping: bar.goingRight ? Theme.springTrail.damping : Theme.springLead.damping
        }
    }
    Behavior on edgeR {
        id: behR
        SpringAnimation {
            spring: bar.goingRight ? Theme.springLead.spring : Theme.springTrail.spring
            damping: bar.goingRight ? Theme.springLead.damping : Theme.springTrail.damping
        }
    }
    SequentialAnimation {
        id: jumpFade
        NumberAnimation { target: indicator; property: "opacity"; to: 0; duration: 0 }
        NumberAnimation { target: indicator; property: "opacity"; to: 1; duration: Theme.motionDuration; easing.type: Easing.OutCubic }
    }

    background: Item {
        implicitHeight: 36

        Item {
            id: trackLayer
            anchors.fill: parent
            GlassSurface {
                anchors.fill: parent
                variant: "card"
                fillColor: Theme.fillTrack
                radius: height / 2
                squircle: 2
            }
            Repeater {
                model: bar.model
                Item {
                    id: seg
                    required property int index
                    required property string modelData
                    x: index * bar.segW
                    width: bar.segW
                    height: trackLayer.height
                    Text {
                        anchors.centerIn: parent
                        width: parent.width - 8
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideRight
                        text: seg.modelData
                        font.family: bar.font.family
                        font.pointSize: bar.font.pointSize
                        font.weight: seg.index === bar.currentIndex ? 600 : 400
                        color: seg.index === bar.currentIndex ? Theme.ink : Theme.inkSecondary
                        Behavior on color { ColorAnimation { duration: Theme.fade } }
                        // in solid / HC the pill is opaque: the selected label is drawn above it instead
                        visible: seg.index !== bar.currentIndex || Theme.mode === "glass"
                    }
                }
            }
        }
        ShaderEffectSource {
            id: trackSrc
            sourceItem: trackLayer
            hideSource: false
            live: true
            smooth: true
            textureSize: Qt.size(trackLayer.width * bar.dpr, trackLayer.height * bar.dpr)
        }
        GlassShadow {
            x: indicator.x; y: indicator.y; width: indicator.width; height: indicator.height
            radius: indicator.radius
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.18)
            opacity: indicator.opacity
        }
        GlassSurface {
            id: indicator
            variant: "pill"
            x: Math.min(bar.edgeL, bar.edgeR)
            y: bar.inset
            width: Math.max(4, Math.abs(bar.edgeR - bar.edgeL))
            height: parent.height - 2 * bar.inset
            radius: Theme.concentric(parent.height / 2, bar.inset)
            squircle: 2
            sourceItem: trackLayer
            sharpSource: trackSrc
        }
        Text {
            visible: Theme.mode !== "glass"
            x: indicator.x + (indicator.width - width) / 2
            y: indicator.y + (indicator.height - height) / 2
            width: Math.min(implicitWidth, indicator.width - 8)
            elide: Text.ElideRight
            text: bar.currentIndex >= 0 && bar.currentIndex < bar.model.length ? bar.model[bar.currentIndex] : ""
            font.family: bar.font.family
            font.pointSize: bar.font.pointSize
            font.weight: 600
            color: Theme.ink
        }
        FocusRing { target: indicator; ringRadius: indicator.radius; active: bar.visualFocus }
    }

    contentItem: ListView {
        model: bar.contentModel
        currentIndex: bar.currentIndex
        orientation: ListView.Horizontal
        interactive: false
        spacing: 0
        clip: false
    }

    // invisible hit / accessibility targets over the track (the labels live in the lensed layer)
    Repeater {
        model: bar.model
        T.TabButton {
            id: tab
            required property int index
            required property string modelData
            width: bar.segW
            // height comes from the bar's contentHeight (see above)
            text: modelData
            focusPolicy: Qt.NoFocus
            hoverEnabled: true
            Accessible.role: Accessible.PageTab
            Accessible.name: text
            background: null
            contentItem: Item {}
            onClicked: {
                bar.currentIndex = tab.index
                bar.activated(tab.index)
            }
        }
    }
}
