import {Component, EventEmitter, Input, Output} from '@angular/core';
import {Paginator} from "../../../paginator";
import {Img} from "../../../models";
import {Location} from "@angular/common";
import {ImgClassifyService} from "./img-classify.service";
import {MatButtonToggleChange} from "@angular/material";
import {DynamicresourceService} from "../../../dynamicresource.service";

@Component({
    selector: 'app-img-classify',
    templateUrl: './img-classify.component.html',
    styleUrls: ['./img-classify.component.css']
})
export class ImgClassifyComponent extends Paginator<Img> {
    protected displayed_columns: string[] = ['img'];
    @Input() task: number;
    @Input() part: string;
    @Input() epochs: number[];
    @Input() group: string;
    @Output() loaded = new EventEmitter<number>();
    epoch: number;
    y: number;
    y_pred: number;
    metric_diff_min: number = 0;
    metric_diff_max: number = 1;

    constructor(protected service: ImgClassifyService, protected location: Location,
                protected resource_service: DynamicresourceService) {
        super(service, location, false);
    }

    protected _ngOnInit() {
        let self = this;
        this.data_updated.subscribe(res => {
            self.resource_service.load('plotly').then(() => {
                self.plot_confusion(res.confusion, res.part, res.class_names);
            });
        });
    }

    plot_confusion(confusion, part, class_names) {
        if (!confusion) {
            return;
        }
        let self = this;
        let colorscaleValue = [
            [0, '#3D9970'],
            [1, '#001f3f']
        ];
        let x = class_names.slice();
        let y = class_names.slice();
        y.reverse();
        confusion.data.reverse();

        let data = [{
            x: x,
            y: y,
            z: confusion.data,
            type: 'heatmap',
            colorscale: colorscaleValue,
            showscale: false
        }];

        let layout = {
            title: 'Heatmap',
            annotations: [],
            width: 300,
            height: 300,
            modebar: false,
            xaxis: {
                ticks: '',
                side: 'top'
            },
            yaxis: {
                ticks: '',
                ticksuffix: ' ',
                width: 700,
                height: 700,
                autosize: false
            }
        };

        for (var i = 0; i < y.length; i++) {
            for (var j = 0; j < x.length; j++) {
                var currentValue = confusion.data[i][j];
                if (currentValue != 0.0) {
                    var textColor = 'white';
                } else {
                    var textColor = 'black';
                }
                var result = {
                    xref: 'x1',
                    yref: 'y1',
                    x: x[j],
                    y: y[i],
                    text: confusion.data[i][j],
                    font: {
                        family: 'Arial',
                        size: 12,
                        color: textColor
                    },
                    showarrow: false,
                };
                layout.annotations.push(result);
            }
        }

        let id = 'img_classify_' + part;
        window['Plotly'].newPlot(id, data, layout, {displayModeBar: false});
        let plot = document.getElementById(id);
        // @ts-ignore
        plot.on('plotly_click', function (data) {
            let pt = data.points[0];
            let y = class_names.indexOf(pt.y);
            let y_pred = class_names.indexOf(pt.x);
            if (y == self.y && y_pred == self.y_pred) {
                self.y = null;
                self.y_pred = null;
            } else {
                self.y = y;
                self.y_pred = y_pred;
            }

            self.change.emit();
        });
    }


    get_filter(): any {
        if (this.epoch == null) {
            return null;
        }
        let res = {};
        res['paginator'] = super.get_filter();
        res['task'] = this.task;
        res['part'] = this.part;
        res['epoch'] = this.epoch;
        res['group'] = this.group;
        res['y'] = this.y;
        res['y_pred'] = this.y_pred;
        res['metric_diff_min'] = this.metric_diff_min;
        res['metric_diff_max'] = this.metric_diff_max;
        return res;
    }

    epoch_changed($event: MatButtonToggleChange) {
        this.epoch = $event.value;
        this.change.emit();
    }
}
