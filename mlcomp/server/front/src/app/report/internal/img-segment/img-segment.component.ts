import {Component, EventEmitter, Input, Output} from '@angular/core';
import {Paginator} from "../../../paginator";
import {Img, ImgClassify, ReportItem} from "../../../models";
import {Location} from "@angular/common";
import {ImgSegmentService} from "./img-segment.service";
import {DynamicresourceService} from "../../../dynamicresource.service";
import {LayoutService} from "../layout/layout.service";

@Component({
    selector: 'app-img-segment',
    templateUrl: './img-segment.component.html',
    styleUrls: ['./img-segment.component.css']
})
export class ImgSegmentComponent extends Paginator<Img> {
    protected displayed_columns: string[] = ['img'];
    @Input() item: ReportItem;
    @Input() data: ImgClassify;

    @Output() loaded = new EventEmitter<number>();
    score_min: number = 0;
    score_max: number = 1;

    constructor(protected service: ImgSegmentService,
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
        res['score_min'] = this.score_min;
        res['score_max'] = this.score_max;
        res['layout'] = this.item;

        return res;
    }
}
