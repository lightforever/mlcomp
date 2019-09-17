import {Component, EventEmitter, Input, Output} from '@angular/core';
import {Paginator} from "../../../paginator";
import {Img, ImgClassify, ReportItem} from "../../../models";
import {Location} from "@angular/common";
import {ImgClassifyService} from "./img-classify.service";
import {DynamicresourceService} from "../../../dynamicresource.service";
import {LayoutService} from "../layout/layout.service";

@Component({
    selector: 'app-img-classify',
    templateUrl: './img-classify.component.html',
    styleUrls: ['./img-classify.component.css']
})
export class ImgClassifyComponent extends Paginator<Img> {
    protected displayed_columns: string[] = ['img'];
    @Input() item: ReportItem;
    @Input() data: ImgClassify;

    @Output() loaded = new EventEmitter<number>();
    y: number;
    y_pred: number;
    score_min: number = 0;
    score_max: number = 1;

    constructor(protected service: ImgClassifyService,
                protected location: Location,
                protected layout_service: LayoutService,
                protected resource_service: DynamicresourceService) {
        super(service,
            location,
            null,
            null,
            false);
    }

    private subscribe_data_changed() {
        this.layout_service.data_updated.subscribe(event => {
            if (event.key != this.item.source) {
                return;
            }
        });
    }

    protected _ngOnInit() {
        let self = this;
        this.subscribe_data_changed();
        this.data_updated.subscribe(res => {
            if(!res || !self.data.name){
                return;
            }
            self.resource_service.load('plotly').
            then(() => {
                self.plot_confusion(res.confusion,
                    self.data.name,
                    res.class_names);
            });
        });
    }

    plot_confusion(confusion, name, class_names) {
        if (!confusion||!confusion.data) {
            return;
        }
        let self = this;
        let colorscaleValue = [
            [0, '#3D9970'],
            [1, '#001f3f']
        ];
        let x = class_names.slice();
        let y = class_names.slice();

        let data = [{
            x: x,
            y: y,
            z: confusion.data,
            type: 'heatmap',
            colorscale: colorscaleValue,
            showscale: false
        }];

        let layout = {
            title: 'Confusion matrix',
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
                autosize: false,
                autorange: 'reversed'
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

        let id = 'img_classify_' + name;
        window['Plotly'].newPlot(id, data, layout, {displayModeBar: false});
        let plot = document.getElementById(id);
        // @ts-ignore
        plot.on('plotly_click', function (data) {
            let pt = data.points[0];
            let y = class_names.indexOf(String(pt.y));
            let y_pred = class_names.indexOf(String(pt.x));
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
        if (!this.data) {
            return null;
        }
        let res = {};
        res['paginator'] = super.get_filter();
        res['task'] = this.data.task;
        res['group'] = this.data.group;
        res['y'] = this.y;
        res['y_pred'] = this.y_pred;
        res['score_min'] = this.score_min;
        res['score_max'] = this.score_max;
        res['layout'] = this.item;

        return res;
    }
}
