import {Component, Input, OnInit} from '@angular/core';
import {ReportItem} from "../../../models";
import {AppSettings} from "../../../app-settings";

@Component({
    selector: 'app-layout',
    templateUrl: './layout.component.html',
    styleUrls: ['./layout.component.css']
})
export class LayoutComponent implements OnInit {

    @Input() item: ReportItem;
    @Input() data;
    id: string = 'series'+this.randid();

    get item_data() {
        return this.data[this.item.source];
    }

    constructor() {
    }

    ngOnInit() {
        if (this.item.type == 'series') {
            this.series();
        }
    }

    randid() {
        return Math.random().toString();
    }

    private series() {
        setTimeout(() => {
            if (document.getElementById(this.id)) {
                let data = this.data[this.item.source];
                for (let plot of data) {
                    let layout = {
                        'title': plot.name,
                        'height': 300,
                        'width': 600,
                        'margin': {'b': 40, 't': 40}
                    };
                    for (let row of plot.data) {
                        let text = [];
                        for (let time of row.time) {
                            time = new Date(Date.parse(time));
                            text.push(AppSettings.format_date_time(time));
                        }

                        row.text = text;
                    }
                    window['Plotly'].newPlot(this.id, plot.data, layout);
                }

            }
        }, 100);
    }
}
