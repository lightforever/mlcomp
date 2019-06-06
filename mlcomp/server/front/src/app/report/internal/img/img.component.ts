import {Component, Input, OnInit} from '@angular/core';
import {ReportItem} from "../../../models";
import {LayoutService} from "../layout/layout.service";

@Component({
    selector: 'app-img',
    templateUrl: './img.component.html',
    styleUrls: ['./img.component.css']
})
export class ImgComponent implements OnInit {

    @Input() item: ReportItem;
    @Input() data;

    constructor(protected layout_service: LayoutService) {
    }

    ngOnInit() {
        this.subscribe_data_changed();
    }


    private subscribe_data_changed() {
        this.layout_service.data_updated.subscribe(event => {
            if (event.key != this.item.source) {
                return;
            }

            this.data = event.data[this.item.index];

        });
    }

}
