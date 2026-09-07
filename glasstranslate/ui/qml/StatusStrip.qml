pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    StatusStrip - pinned at the bottom of every tab (docs/GLASS_DESIGN.md §2.3):
    a "fill" bar (radius concentric with the slab: 22 - 14 = 8) with the status
    message on the left and, on the right, the running/paused dot, the
    `Total x ms · y fps` readout in Theme.mono and a chevron that toggles the
    details drawer.  The drawer is a GlassCard above the bar with the four
    read-only, selectable detail lines (Stages, Segments, Cache, Devices); its
    height springs open (120 ms OutCubic under reduceMotion).  implicitHeight
    includes the drawer, so the page area above shrinks while it is open.
*/
Item {
    id: strip

    property string message: "Ready"
    property string totalText: "-"
    property string fpsText: "-"
    property string stagesText: "-"
    property string segmentsText: "-"
    property string cacheText: "-"
    property string devicesText: "-"
    property bool running: false
    property bool detailsOpen: false

    readonly property real barRadius: Theme.concentric(Theme.radiusWindow, Theme.pad)
    readonly property real barH: Math.max(Theme.stripH, fm.height + 2 * Theme.padY + 8)
    // bridge.stats.totalText already carries its unit ("12.3 ms"), fpsText does not ("8.1")
    readonly property string readout: "Total " + totalText + " · " + fpsText + " fps"

    implicitHeight: barH + (drawer.height > 0 ? drawer.height + Theme.gap : 0)
    Accessible.role: Accessible.StatusBar
    Accessible.name: "Status"

    FontMetrics { id: fm; font.family: Theme.family; font.pointSize: Theme.fontBody }

    // ---- details drawer (above the bar)
    GlassCard {
        id: drawer
        objectName: "detailsDrawer"
        anchors.left: parent.left
        anchors.right: parent.right
        y: 0
        clip: true
        height: strip.detailsOpen ? implicitHeight : 0
        visible: height > 0.5
        Behavior on height { animation: Theme.reduceMotion ? drawerEase : drawerSpring }

        Column {
            width: parent.width
            spacing: 4
            Repeater {
                model: [
                    { key: "Stages (ms)", text: strip.stagesText },
                    { key: "Segments", text: strip.segmentsText },
                    { key: "Cache", text: strip.cacheText },
                    { key: "Devices", text: strip.devicesText }
                ]
                Item {
                    id: line
                    required property int index
                    required property var modelData
                    width: parent.width
                    height: Math.max(keyText.implicitHeight, valueEdit.implicitHeight)
                    Text {
                        id: keyText
                        width: Math.round(96 * Theme.textScale)
                        text: line.modelData.key
                        font.family: Theme.family
                        font.pointSize: Theme.fontCaption
                        color: Theme.inkSecondary
                    }
                    TextEdit {
                        id: valueEdit
                        x: keyText.width + Theme.gap
                        width: parent.width - x
                        text: line.modelData.text
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        font.family: Theme.mono
                        font.pointSize: Theme.fontCaption
                        color: Theme.ink
                        selectionColor: Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, 0.25)
                        selectedTextColor: Theme.ink
                        Accessible.role: Accessible.StaticText
                        Accessible.name: line.modelData.key + ": " + text
                    }
                }
            }
        }
    }
    NumberAnimation { id: drawerEase; duration: Theme.motionDuration; easing.type: Easing.OutCubic }
    SpringAnimation { id: drawerSpring; spring: Theme.springLead.spring; damping: Theme.springLead.damping }

    // ---- the bar
    Item {
        id: bar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: strip.barH

        GlassShadow {
            anchors.fill: parent
            radius: strip.barRadius
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.16)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "fill"
            radius: strip.barRadius
        }

        Text {
            id: messageText
            objectName: "statusMessage"
            anchors.left: parent.left
            anchors.leftMargin: Theme.pad
            anchors.right: right.left
            anchors.rightMargin: Theme.gap
            anchors.verticalCenter: parent.verticalCenter
            text: strip.message
            elide: Text.ElideRight
            font.family: Theme.family
            font.pointSize: Theme.fontBody
            color: Theme.ink
        }

        Row {
            id: right
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.verticalCenter: parent.verticalCenter
            spacing: Theme.gap
            Rectangle {
                width: 6
                height: 6
                radius: 3
                anchors.verticalCenter: parent.verticalCenter
                color: strip.running ? Theme.accent : Qt.rgba(Theme.inkSecondary.r, Theme.inkSecondary.g, Theme.inkSecondary.b, 0.5)
                Accessible.role: Accessible.Indicator
                Accessible.name: strip.running ? "Running" : "Paused"
            }
            Text {
                objectName: "readout"
                anchors.verticalCenter: parent.verticalCenter
                text: strip.readout
                font.family: Theme.mono
                font.pointSize: Theme.fontBody
                color: Theme.ink
            }
            T.Button {
                id: chevron
                objectName: "detailsButton"
                width: 28
                height: 28
                anchors.verticalCenter: parent.verticalCenter
                focusPolicy: Qt.StrongFocus
                hoverEnabled: true
                checkable: true
                checked: strip.detailsOpen
                onClicked: strip.detailsOpen = !strip.detailsOpen
                Accessible.role: Accessible.Button
                Accessible.name: strip.detailsOpen ? "Hide details" : "Show details"
                background: Item {
                    FocusRing { ringRadius: 8; active: chevron.visualFocus }
                }
                contentItem: Glyph {
                    name: "chevron-up"
                    color: Theme.ink
                    rotation: strip.detailsOpen ? 180 : 0
                    Behavior on rotation { NumberAnimation { duration: Theme.fade; easing.type: Easing.OutCubic } }
                }
            }
        }
    }
}
