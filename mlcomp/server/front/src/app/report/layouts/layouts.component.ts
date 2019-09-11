import {Component, ViewChild} from '@angular/core';
import {Paginator} from "../../paginator";
import {Layout} from "../../models";
import {Location} from "@angular/common";
import {LayoutsService} from "./layouts.service";
import {MatDialog} from "@angular/material";
import {LayoutAddDialogComponent} from "./layout-add-dialog";
import {el} from "@angular/platform-browser/testing/src/browser_util";
import {Helpers} from "../../helpers";

@Component({
    selector: 'app-layouts',
    templateUrl: './layouts.component.html',
    styleUrls: ['./layouts.component.css']
})
export class LayoutsComponent extends Paginator<Layout> {
    protected displayed_columns: string[] = [
        'name',
        'last_modified'
    ];

    id_column = 'name';
    selected: Layout;
    error: string;
    @ViewChild('textarea') textarea;

    row_select(element: Layout) {
        this.selected = element;
        this.textarea.nativeElement.value = element.content;
    };

    constructor(
        protected service: LayoutsService,
        protected location: Location,
        public dialog: MatDialog
    ) {
        super(service, location,
            null,
            null,
            false);
    }

    get_filter(): any {
        let res = {};
        res['paginator'] = super.get_filter();
        res['paginator']['sort_column'] = 'last_modified';
        return res;
    }

    add() {
        const dialogRef = this.dialog.open(LayoutAddDialogComponent,
            {
                width: '400px', height: '200px',
                data: {}
            });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.add(result).subscribe(res => {
                    this.change.emit();

                    this.error = res.error;
                });
            }
        });
    }

    edit_name() {
        let name = this.selected.name;
        const dialogRef = this.dialog.open(LayoutAddDialogComponent,
            {
                width: '400px', height: '200px',
                data: {'name': name}
            });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.edit(name, null, result.name).subscribe(
                    res => {
                        this.change.emit();
                        this.error = res.error;
                    });

            }
        });
    }

    remove() {
        this.service.remove(this.selected.name)
            .subscribe(res => {
                    if (res.success) {
                        this.change.emit();
                        this.selected = null;
                    }
                    this.error = res.error;
                },
            );
    }

    save() {
        this.service.edit(this.selected.name, this.selected.content)
            .subscribe(res => {
                    if (res.success) {
                        this.change.emit();
                    }
                    this.error = res.error;
                },
            );
    }

    key_down(event) {
        if (!this.selected) {
            return;
        }

        let content = Helpers.handle_textarea_down_key(event,
            this.textarea.nativeElement);
        if (content) {
            this.selected.content = content;
        }

    }

    key_up(event) {
        if (!this.selected) {
            return;
        }

        this.selected.content = event.target.value;
    }

}
